// 直接在 pack（{split}.bin + .offsets.npy + .labels.npy）上做 SimHash 近重复去重，并原地重写。
// 用法: dedup-pack <root> <并发> [--keep-cross-class]

using System.Diagnostics;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

internal static class PackDedup
{
    // 流式读写 .bin 时的分块大小（按文档数切，保证块边界落在文档边界上）。
    private const int ChunkDocs = 16384;
    private static readonly string[] Splits = { "train", "val", "test" };

    // dropCrossClass：跨类近重复簇整簇丢弃（默认，去掉了标签自相矛盾的样本）；保持 false 则按普通簇处理。
    // dryRun：只算不写，用来核对去重量。
    public static int Run(string root, int workers, bool dropCrossClass = true, bool dryRun = false)
    {
        if (workers <= 0)
        {
            Console.Error.WriteLine($"并发数必须 >= 1，收到 {workers}");
            return 1;
        }

        var packDir = Path.Combine(root, "packs");
        var infoPath = Path.Combine(root, "dataset_info.json");
        var info = JsonNode.Parse(File.ReadAllText(infoPath))
                   ?? throw new InvalidDataException($"解析失败: {infoPath}");
        var classNames = info["class_names"]!.AsArray().Select(node => node!.GetValue<string>()).ToArray();

        // 只把 offsets/labels 读进内存；.bin 全程分块流式读写，峰值内存与 pack 总大小无关。
        var offsets = new long[Splits.Length][];
        var labels = new long[Splits.Length][];
        for (var split = 0; split < Splits.Length; split++)
        {
            offsets[split] = Npy.ReadInt64(Path.Combine(packDir, $"{Splits[split]}.offsets.npy"));
            labels[split] = Npy.ReadInt64(Path.Combine(packDir, $"{Splits[split]}.labels.npy"));
            if (offsets[split].Length != labels[split].Length + 1)
            {
                throw new InvalidDataException(
                    $"{Splits[split]}.offsets.npy 有 {offsets[split].Length} 项、labels 有 {labels[split].Length} 项，应满足 offsets = labels + 1");
            }
        }

        var starts = new int[Splits.Length];
        var total = 0;
        for (var split = 0; split < Splits.Length; split++)
        {
            starts[split] = total;
            total += labels[split].Length;
        }

        var docSplit = new byte[total];
        var docClass = new int[total];
        for (var split = 0; split < Splits.Length; split++)
        {
            for (var i = 0; i < labels[split].Length; i++)
            {
                var index = starts[split] + i;
                var label = labels[split][i];
                if (label < 0 || label >= classNames.Length)
                {
                    throw new InvalidDataException($"{Splits[split]}.labels.npy 第 {i} 项标签越界: {label}");
                }

                docSplit[index] = (byte)split;
                docClass[index] = (int)label;
            }
        }

        var timer = Stopwatch.StartNew();
        var hashes = new ulong[total];
        for (var split = 0; split < Splits.Length; split++)
        {
            HashSplit(Path.Combine(packDir, $"{Splits[split]}.bin"), offsets[split], starts[split], hashes, workers);
        }

        Console.WriteLine($"SimHash 完成: {total} 篇, {timer.Elapsed.TotalSeconds:F1}s");

        var parent = Lsh.BuildClusters(hashes, out var lshStats);
        Console.WriteLine(
            $"近重复候选比较: {lshStats.Candidates:N0} 次（汉明距离 <= {Lsh.MaxDistance}），完全重复 {lshStats.IdenticalPairs:N0} 对");

        var best = new int[total];
        Array.Fill(best, -1);
        for (var index = 0; index < total; index++)
        {
            var rootIndex = UnionFind.Find(parent, index);
            if (best[rootIndex] < 0 || Better(index, best[rootIndex], docSplit))
            {
                best[rootIndex] = index;
            }
        }

        var keep = new bool[total];
        var clusters = 0;
        for (var index = 0; index < total; index++)
        {
            if (best[index] >= 0)
            {
                keep[best[index]] = true;
                clusters++;
            }
        }

        var crossClass = AnalyzeCrossClass(parent, docClass);
        if (dropCrossClass)
        {
            // 跨类簇一篇都不留：簇内文档本来就要删到只剩一篇，现在连那篇也不进 pack。
            for (var index = 0; index < total; index++)
            {
                if (crossClass.Mask[index])
                {
                    keep[index] = false;
                }
            }
        }

        Console.WriteLine(
            $"连通分量: {clusters}，保留 {keep.Count(v => v)} / {total}，跨类簇 {crossClass.Clusters} 个（涉及 {crossClass.Documents} 篇）"
            + (dropCrossClass ? "，已整簇丢弃" : "，保留"));

        if (dryRun)
        {
            for (var split = 0; split < Splits.Length; split++)
            {
                var keptCount = 0;
                for (var i = 0; i < labels[split].Length; i++)
                {
                    if (keep[starts[split] + i])
                    {
                        keptCount++;
                    }
                }

                Console.WriteLine($"  {Splits[split],-5}: {labels[split].Length} -> {keptCount}（去掉 {labels[split].Length - keptCount}）");
            }

            Console.WriteLine($"共去掉 {total - keep.Count(v => v)} 篇；dry-run，未写入任何文件");
            return 0;
        }

        // 写新的 pack
        var newDir = Path.Combine(root, "packs.new");
        if (Directory.Exists(newDir))
        {
            Directory.Delete(newDir, true);
        }

        Directory.CreateDirectory(newDir);
        var kept = new long[Splits.Length];
        var perClass = new long[Splits.Length][];
        for (var split = 0; split < Splits.Length; split++)
        {
            perClass[split] = new long[classNames.Length];
            var newOffsets = new List<long>(labels[split].Length) { 0 };
            var newLabels = new List<long>(labels[split].Length);
            long position = 0;
            var count = labels[split].Length;
            using (var input = new FileStream(
                       Path.Combine(packDir, $"{Splits[split]}.bin"),
                       FileMode.Open,
                       FileAccess.Read,
                       FileShare.Read,
                       1 << 20,
                       FileOptions.SequentialScan))
            using (var output = new FileStream(
                       Path.Combine(newDir, $"{Splits[split]}.bin"),
                       FileMode.Create,
                       FileAccess.Write,
                       FileShare.None,
                       1 << 20))
            {
                for (var first = 0; first < count; first += ChunkDocs)
                {
                    var chunkCount = Math.Min(ChunkDocs, count - first);
                    var byteStart = offsets[split][first];
                    var buffer = new byte[checked((int)(offsets[split][first + chunkCount] - byteStart))];
                    input.Position = byteStart;
                    input.ReadExactly(buffer);
                    for (var k = 0; k < chunkCount; k++)
                    {
                        var index = starts[split] + first + k;
                        if (!keep[index])
                        {
                            continue;
                        }

                        var docStart = checked((int)(offsets[split][first + k] - byteStart));
                        var docLength = checked((int)(offsets[split][first + k + 1] - offsets[split][first + k]));
                        output.Write(buffer, docStart, docLength);
                        position += docLength;
                        newOffsets.Add(position);
                        newLabels.Add(labels[split][first + k]);
                        perClass[split][labels[split][first + k]]++;
                    }
                }
            }

            Npy.WriteInt64(Path.Combine(newDir, $"{Splits[split]}.offsets.npy"), newOffsets.ToArray());
            Npy.WriteInt64(Path.Combine(newDir, $"{Splits[split]}.labels.npy"), newLabels.ToArray());
            kept[split] = newLabels.Count;
        }

        // 交换目录：第二步失败就把旧目录挪回去，避免“packs 不见了、只剩 packs.old”。
        var oldDir = Path.Combine(root, "packs.old");
        if (Directory.Exists(oldDir))
        {
            Directory.Delete(oldDir, true);
        }

        Directory.Move(packDir, oldDir);
        try
        {
            Directory.Move(newDir, packDir);
        }
        catch
        {
            Directory.Move(oldDir, packDir);
            throw;
        }

        Directory.Delete(oldDir, true);

        // 更新 dataset_info.json
        for (var split = 0; split < Splits.Length; split++)
        {
            var section = info["splits"]![Splits[split]]!;
            section["total"] = kept[split];
            var classes = new JsonObject();
            for (var label = 0; label < classNames.Length; label++)
            {
                classes[classNames[label]] = perClass[split][label];
            }

            section["per_class"] = classes;
        }

        var dropped = total - kept.Sum();
        info["dedup"] = new JsonObject
        {
            ["enabled"] = true,
            ["method"] = $"simhash64(字符3-gram, 唯一 trigram 权重)，汉明距离 <= {Lsh.MaxDistance}"
                         + "（16-bit×7 窗口 + 1-bit 邻桶多探针，覆盖完整）",
            ["policy"] = dropCrossClass
                ? "每个近重复簇保留一篇（train 优先，其次编号最小）；跨类簇整簇丢弃"
                : "每个近重复簇保留一篇：train 优先，其次编号最小",
            ["candidates"] = lshStats.Candidates,
            ["identical_pairs"] = lshStats.IdenticalPairs,
            ["clusters"] = clusters,
            ["dropped"] = dropped,
            ["cross_class"] = new JsonObject
            {
                ["policy"] = dropCrossClass ? "drop-cluster" : "keep",
                ["clusters"] = crossClass.Clusters,
                ["documents"] = crossClass.Documents,
            },
        };
        File.WriteAllText(
            infoPath,
            info.ToJsonString(new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            }) + "\n");

        for (var split = 0; split < Splits.Length; split++)
        {
            Console.WriteLine($"  {Splits[split],-5}: {labels[split].Length} -> {kept[split]}（去掉 {labels[split].Length - kept[split]}）");
        }

        Console.WriteLine($"共去掉 {dropped} 篇，总耗时 {timer.Elapsed.TotalSeconds:F1}s");
        return 0;
    }

    /// <summary>逐块读一个 split 的 .bin，算好每篇的 SimHash，写进 <paramref name="hashes"/> 的对应区间。</summary>
    private static void HashSplit(string binPath, long[] offsets, int targetBase, ulong[] hashes, int workers)
    {
        var count = offsets.Length - 1;
        using var stream = new FileStream(
            binPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        for (var first = 0; first < count; first += ChunkDocs)
        {
            var chunkCount = Math.Min(ChunkDocs, count - first);
            var byteStart = offsets[first];
            var buffer = new byte[checked((int)(offsets[first + chunkCount] - byteStart))];
            stream.Position = byteStart;
            stream.ReadExactly(buffer);
            var chunkBase = first;
            Parallel.For(
                0,
                chunkCount,
                new ParallelOptions { MaxDegreeOfParallelism = workers },
                () => new Program.LocalState(),
                (k, _, state) =>
                {
                    var begin = checked((int)(offsets[chunkBase + k] - byteStart));
                    var length = checked((int)(offsets[chunkBase + k + 1] - offsets[chunkBase + k]));
                    hashes[targetBase + chunkBase + k] = Program.SimHash(buffer.AsSpan(begin, length), state);
                    return state;
                },
                _ => { });
        }
    }

    /// <summary>
    /// 找出「跨类重复」：一个近重复簇里出现 ≥2 个不同分类时算一个跨类簇，簇里全部文档都算跨类重复文档
    /// （包括本来会被保留的那篇 —— 它和别的分类里的稿子重复，本身就是跨类的重复证据）。
    /// </summary>
    /// <param name="parent">并查集数组（由 <see cref="Lsh.BuildClusters"/> 得到的结果）。</param>
    /// <param name="groups">每篇文档的分类下标，长度必须与 <paramref name="parent"/> 相同。</param>
    /// <returns>每篇文档是否属于跨类簇、跨类簇数、这些簇里的文档数。</returns>
    internal static (bool[] Mask, long Clusters, long Documents) AnalyzeCrossClass(int[] parent, int[] groups)
    {
        // 只记每个簇第一次见到的分类：再见到别的分类就把该簇标成跨类，不用给每个簇存集合。
        var firstClass = new int[groups.Length];
        Array.Fill(firstClass, -1);
        var crossRoot = new bool[groups.Length];
        for (var index = 0; index < groups.Length; index++)
        {
            var root = UnionFind.Find(parent, index);
            if (firstClass[root] < 0)
            {
                firstClass[root] = groups[index];
            }
            else if (!crossRoot[root] && firstClass[root] != groups[index])
            {
                crossRoot[root] = true;
            }
        }

        var mask = new bool[groups.Length];
        long clusters = 0;
        long documents = 0;
        for (var index = 0; index < groups.Length; index++)
        {
            if (crossRoot[index])
            {
                clusters++;
            }

            if (crossRoot[UnionFind.Find(parent, index)])
            {
                mask[index] = true;
                documents++;
            }
        }

        return (mask, clusters, documents);
    }

    private static bool Better(int candidate, int current, byte[] docSplit)
    {
        if (docSplit[candidate] != docSplit[current])
        {
            return docSplit[candidate] < docSplit[current];
        }

        return candidate < current;
    }
}
