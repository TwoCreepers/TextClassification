// 直接在 pack（{split}.bin + .offsets.npy + .labels.npy）上做 SimHash 近重复去重，并原地重写。
// 用法: dedup-pack <root> <等待并发>

using System.Numerics;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

internal static class PackDedup
{
    private const int MaxDistance = 6;
    private static readonly string[] Splits = { "train", "val", "test" };

    public static int Run(string root, int workers)
    {
        var packDir = Path.Combine(root, "packs");
        var infoPath = Path.Combine(root, "dataset_info.json");
        var info = JsonNode.Parse(File.ReadAllText(infoPath))
                   ?? throw new InvalidDataException($"解析失败: {infoPath}");
        var classNames = info["class_names"]!.AsArray().Select(node => node!.GetValue<string>()).ToArray();

        var bins = new byte[3][];
        var offsets = new long[3][];
        var labels = new long[3][];
        for (var split = 0; split < Splits.Length; split++)
        {
            bins[split] = File.ReadAllBytes(Path.Combine(packDir, $"{Splits[split]}.bin"));
            offsets[split] = Npy.ReadInt64(Path.Combine(packDir, $"{Splits[split]}.offsets.npy"));
            labels[split] = Npy.ReadInt64(Path.Combine(packDir, $"{Splits[split]}.labels.npy"));
        }

        var starts = new int[3];
        var total = 0;
        for (var split = 0; split < Splits.Length; split++)
        {
            starts[split] = total;
            total += labels[split].Length;
        }

        var docSplit = new byte[total];
        var docClass = new int[total];
        var docStart = new int[total];
        var docLength = new int[total];
        for (var split = 0; split < Splits.Length; split++)
        {
            for (var i = 0; i < labels[split].Length; i++)
            {
                var index = starts[split] + i;
                docSplit[index] = (byte)split;
                docClass[index] = (int)labels[split][i];
                docStart[index] = (int)offsets[split][i];
                docLength[index] = (int)(offsets[split][i + 1] - offsets[split][i]);
            }
        }

        var timer = System.Diagnostics.Stopwatch.StartNew();
        var hashes = new ulong[total];
        Parallel.For(
            0,
            total,
            new ParallelOptions { MaxDegreeOfParallelism = workers },
            () => new Program.LocalState(),
            (index, _, state) =>
            {
                hashes[index] = Program.SimHash(
                    Encoding.UTF8.GetString(bins[docSplit[index]], docStart[index], docLength[index]),
                    state);
                return state;
            },
            _ => { });
        Console.WriteLine($"SimHash 完成: {total} 篇, {timer.Elapsed.TotalSeconds:F1}s");

        var pairs = BuildPairs(hashes);
        Console.WriteLine($"近重复对（汉明距离 <= {MaxDistance}）: {pairs.Count}");

        var parent = new int[total];
        for (var i = 0; i < total; i++)
        {
            parent[i] = i;
        }

        foreach (var pair in pairs)
        {
            Union(parent, (int)(pair >> 32), (int)(pair & 0xFFFFFFFF));
        }

        var best = new int[total];
        Array.Fill(best, -1);
        for (var index = 0; index < total; index++)
        {
            var rootIndex = Find(parent, index);
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

        var crossClass = CountCrossClass(parent, docClass);
        Console.WriteLine(
            $"连通分量: {clusters}，保留 {keep.Count(v => v)} / {total}，跨类簇 {crossClass.Clusters} 个（涉及 {crossClass.Documents} 篇）");

        // 写新的 pack
        var newDir = Path.Combine(root, "packs.new");
        if (Directory.Exists(newDir))
        {
            Directory.Delete(newDir, true);
        }

        Directory.CreateDirectory(newDir);
        var kept = new long[3];
        var perClass = new long[3][];
        for (var split = 0; split < Splits.Length; split++)
        {
            perClass[split] = new long[classNames.Length];
            var newOffsets = new List<long>(labels[split].Length) { 0 };
            var newLabels = new List<long>(labels[split].Length);
            long position = 0;
            using (var stream = new FileStream(
                       Path.Combine(newDir, $"{Splits[split]}.bin"),
                       FileMode.Create,
                       FileAccess.Write,
                       FileShare.None,
                       1 << 20))
            {
                for (var i = 0; i < labels[split].Length; i++)
                {
                    var index = starts[split] + i;
                    if (!keep[index])
                    {
                        continue;
                    }

                    stream.Write(bins[split], docStart[index], docLength[index]);
                    position += docLength[index];
                    newOffsets.Add(position);
                    newLabels.Add(labels[split][i]);
                    perClass[split][labels[split][i]]++;
                }
            }

            Npy.WriteInt64(Path.Combine(newDir, $"{Splits[split]}.offsets.npy"), newOffsets.ToArray());
            Npy.WriteInt64(Path.Combine(newDir, $"{Splits[split]}.labels.npy"), newLabels.ToArray());
            kept[split] = newLabels.Count;
        }

        // 交换目录
        var oldDir = Path.Combine(root, "packs.old");
        if (Directory.Exists(oldDir))
        {
            Directory.Delete(oldDir, true);
        }

        Directory.Move(packDir, oldDir);
        Directory.Move(newDir, packDir);
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
            ["method"] = $"simhash64(字符3-gram, 唯一 trigram 权重)，相似度 > 0.9（汉明距离 <= {MaxDistance}）",
            ["policy"] = "每个近重复簇保留一篇：train 优先，其次编号最小",
            ["clusters"] = clusters,
            ["dropped"] = dropped,
            ["cross_class"] = new JsonObject
            {
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

    internal static HashSet<long> BuildPairs(ulong[] hashes)
    {
        const int windowBits = 16;
        const int stride = 8;
        var keys = new ulong[hashes.Length];
        var sorted = new int[hashes.Length];
        var counts = new int[1 << windowBits];
        var pairs = new HashSet<long>();

        for (var offset = 0; offset + windowBits <= 64; offset += stride)
        {
            Array.Clear(counts);
            for (var i = 0; i < hashes.Length; i++)
            {
                var key = (int)((hashes[i] >> offset) & ((1UL << windowBits) - 1));
                keys[i] = (ulong)key;
                counts[key]++;
            }

            var sum = 0;
            for (var k = 0; k < counts.Length; k++)
            {
                var count = counts[k];
                counts[k] = sum;
                sum += count;
            }

            for (var i = 0; i < hashes.Length; i++)
            {
                sorted[counts[keys[i]]++] = i;
            }

            var start = 0;
            for (var k = 0; k < counts.Length; k++)
            {
                var end = k + 1 < counts.Length ? counts[k + 1] : hashes.Length;
                for (var a = start; a < end; a++)
                {
                    for (var b = a + 1; b < end; b++)
                    {
                        var i = sorted[a];
                        var j = sorted[b];
                        if (BitOperations.PopCount(hashes[i] ^ hashes[j]) > MaxDistance)
                        {
                            continue;
                        }

                        var lo = Math.Min(i, j);
                        var hi = Math.Max(i, j);
                        pairs.Add(((long)lo << 32) | (uint)hi);
                    }
                }

                start = end;
            }
        }

        return pairs;
    }

    /// <summary>
    /// 统计「跨类重复」：一个近重复簇里出现 ≥2 个不同分类时算一个跨类簇，簇里的全部文档都算跨类重复文档
    /// （包括被保留的那篇 —— 它和别的分类里的稿子重复，本身就是跨类的重复证据）。
    /// </summary>
    /// <param name="parent">并查集数组（由 <see cref="BuildPairs"/> 得到的结果建出来的那个）。</param>
    /// <param name="groups">每篇文档的分类下标，长度必须与 <paramref name="parent"/> 相同。</param>
    /// <returns>跨类簇数、这些簇里的文档数。</returns>
    internal static (long Clusters, long Documents) CountCrossClass(int[] parent, int[] groups)
    {
        // 只记每个簇第一次见到的分类：再见到别的分类就把该簇标成跨类，不用给每个簇存集合。
        var firstClass = new int[groups.Length];
        Array.Fill(firstClass, -1);
        var cross = new bool[groups.Length];
        for (var index = 0; index < groups.Length; index++)
        {
            var root = Find(parent, index);
            if (firstClass[root] < 0)
            {
                firstClass[root] = groups[index];
            }
            else if (!cross[root] && firstClass[root] != groups[index])
            {
                cross[root] = true;
            }
        }

        long clusters = 0;
        long documents = 0;
        for (var index = 0; index < groups.Length; index++)
        {
            if (cross[index])
            {
                clusters++;
            }

            if (cross[Find(parent, index)])
            {
                documents++;
            }
        }

        return (clusters, documents);
    }

    private static bool Better(int candidate, int current, byte[] docSplit)
    {
        if (docSplit[candidate] != docSplit[current])
        {
            return docSplit[candidate] < docSplit[current];
        }

        return candidate < current;
    }

    private static int Find(int[] parent, int index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }

        return index;
    }

    private static void Union(int[] parent, int a, int b)
    {
        var rootA = Find(parent, a);
        var rootB = Find(parent, b);
        if (rootA != rootB)
        {
            parent[rootB] = rootA;
        }
    }
}
