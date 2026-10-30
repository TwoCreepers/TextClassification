// C# 版预处理：读原始 THUCNews -> SimHash 去重 -> 按分类 8:1:1 划分 -> 写 pack。
// 用法: prep <source> <output> [--seed N] [--limit-per-class N] [--workers N] [--no-dedup] [--overwrite]

using System.Diagnostics;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

internal static class Prep
{
    private const string InfoName = "dataset_info.json";
    private const int MaxDistance = 6;
    private static readonly string[] Splits = { "train", "val", "test" };

    private sealed class SplitMix64
    {
        private ulong _state;

        public SplitMix64(ulong seed)
        {
            _state = seed;
        }

        public ulong Next()
        {
            var z = _state += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    public static int Run(
        string source,
        string output,
        int seed,
        int limitPerClass,
        int workers,
        bool dedup,
        bool overwrite)
    {
        var timer = Stopwatch.StartNew();
        var sourceDir = new DirectoryInfo(source);
        if (!sourceDir.Exists)
        {
            Console.Error.WriteLine($"源目录不存在: {source}");
            return 1;
        }

        if (File.Exists(Path.Combine(output, InfoName)) && !overwrite)
        {
            Console.Error.WriteLine($"{output} 已经处理过（存在 {InfoName}），重新生成请加 --overwrite");
            return 1;
        }

        var classDirs = sourceDir.GetDirectories().OrderBy(dir => dir.Name, StringComparer.Ordinal).ToArray();
        if (classDirs.Length == 0)
        {
            Console.Error.WriteLine($"{source} 下没有任何分类目录");
            return 1;
        }

        // ---- 枚举文档（分类按名字排序，类内按数字 id 排序）----
        var paths = new List<string>();
        var docClass = new List<int>();
        var docSize = new List<int>();
        for (var classIndex = 0; classIndex < classDirs.Length; classIndex++)
        {
            var files = classDirs[classIndex].GetFiles("*.txt");
            Array.Sort(files, CompareNumeric);
            var take = limitPerClass > 0 && files.Length > limitPerClass ? limitPerClass : files.Length;
            for (var i = 0; i < take; i++)
            {
                paths.Add(files[i].FullName);
                docClass.Add(classIndex);
                docSize.Add((int)files[i].Length);
            }
        }

        var count = paths.Count;
        var totalBytes = 0L;
        foreach (var size in docSize)
        {
            totalBytes += size;
        }

        Console.WriteLine(
            $"源: {count} 篇 / {classDirs.Length} 类 / {totalBytes / 1024.0 / 1024:F1} MiB，枚举 {timer.Elapsed.TotalSeconds:F1}s");

        // ---- 一次性读进内存（分块，单块 1 GiB；.NET 单个数组最多 2^31-1 个元素，装不下 2.09 GB）----
        var corpus = CreateChunks(totalBytes);
        var offsets = new long[count + 1];
        for (var i = 0; i < count; i++)
        {
            offsets[i + 1] = offsets[i] + docSize[i];
        }

        Parallel.For(
            0,
            count,
            new ParallelOptions { MaxDegreeOfParallelism = workers },
            index =>
            {
                var buffer = new byte[docSize[index]];
                using (var stream = new FileStream(
                           paths[index],
                           FileMode.Open,
                           FileAccess.Read,
                           FileShare.Read,
                           1,
                           FileOptions.SequentialScan))
                {
                    stream.ReadExactly(buffer);
                }

                CopyInto(corpus, offsets[index], buffer);
            });
        Console.WriteLine($"读入内存: {timer.Elapsed.TotalSeconds:F1}s");

        // ---- SimHash ----
        var hashes = new ulong[count];
        Parallel.For(
            0,
            count,
            new ParallelOptions { MaxDegreeOfParallelism = workers },
            () => new Program.LocalState(),
            (index, _, state) =>
            {
                hashes[index] = Program.SimHash(GetText(corpus, offsets[index], docSize[index]), state);
                return state;
            },
            _ => { });
        Console.WriteLine($"SimHash: {timer.Elapsed.TotalSeconds:F1}s");

        // ---- 去重：每簇保留“分类 + 编号”最小的那篇 ----
        var keep = new bool[count];
        Array.Fill(keep, true);
        var clusters = 0;
        var pairs = 0;
        var crossClass = (Clusters: 0L, Documents: 0L);
        if (dedup)
        {
            var duplicatePairs = PackDedup.BuildPairs(hashes);
            pairs = duplicatePairs.Count;
            var parent = new int[count];
            for (var i = 0; i < count; i++)
            {
                parent[i] = i;
            }

            foreach (var pair in duplicatePairs)
            {
                var a = (int)(pair >> 32);
                var b = (int)(pair & 0xFFFFFFFF);
                var rootA = Find(parent, a);
                var rootB = Find(parent, b);
                if (rootA != rootB)
                {
                    parent[rootB] = rootA;
                }
            }

            var best = new int[count];
            Array.Fill(best, -1);
            for (var i = 0; i < count; i++)
            {
                var root = Find(parent, i);
                if (best[root] < 0 || i < best[root])
                {
                    best[root] = i;
                }
            }

            Array.Fill(keep, false);
            for (var i = 0; i < count; i++)
            {
                if (best[i] >= 0)
                {
                    keep[best[i]] = true;
                    clusters++;
                }
            }

            Console.WriteLine(
                $"去重: 近重复对 {pairs}，簇 {clusters}，保留 {keep.Count(v => v)} / {count}，{timer.Elapsed.TotalSeconds:F1}s");

            // 跨类簇：簇里出现 ≥2 个分类，说明同一篇稿子被挂到了不同分类下（THUCNews 里常见于话题重叠）。
            crossClass = PackDedup.CountCrossClass(parent, docClass.ToArray());
            Console.WriteLine(
                $"跨类重复: {crossClass.Clusters} 簇 / {crossClass.Documents} 篇，{timer.Elapsed.TotalSeconds:F1}s");
        }
        else
        {
            Console.WriteLine("去重: 已跳过（--no-dedup）");
        }

        // ---- 每个分类各自洗牌后 8:1:1 ----
        var splitOf = new byte[count];
        Array.Fill(splitOf, (byte)255);
        var shuffledByClass = new int[classDirs.Length][];
        for (var classIndex = 0; classIndex < classDirs.Length; classIndex++)
        {
            var members = new List<int>();
            for (var i = 0; i < count; i++)
            {
                if (docClass[i] == classIndex && keep[i])
                {
                    members.Add(i);
                }
            }

            var order = members.ToArray();
            var rng = new SplitMix64(SeedFor(seed, classDirs[classIndex].Name));
            for (var i = order.Length - 1; i > 0; i--)
            {
                var j = (int)(rng.Next() % (ulong)(i + 1));
                (order[i], order[j]) = (order[j], order[i]);
            }

            var (train, val, _) = SplitCounts(order.Length);
            for (var i = 0; i < order.Length; i++)
            {
                splitOf[order[i]] = i < train ? (byte)0 : i < train + val ? (byte)1 : (byte)2;
            }

            shuffledByClass[classIndex] = order;
        }

        Console.WriteLine($"划分: {timer.Elapsed.TotalSeconds:F1}s");

        // ---- 写 pack ----
        Directory.CreateDirectory(output);
        var newDir = Path.Combine(output, "packs.new");
        if (Directory.Exists(newDir))
        {
            Directory.Delete(newDir, true);
        }

        Directory.CreateDirectory(newDir);
        var totals = new long[3];
        var perClass = new long[3][];
        for (var split = 0; split < Splits.Length; split++)
        {
            perClass[split] = new long[classDirs.Length];
            var newOffsets = new List<long>(count) { 0 };
            var newLabels = new List<long>(count);
            long position = 0;
            using (var stream = new FileStream(
                       Path.Combine(newDir, $"{Splits[split]}.bin"),
                       FileMode.Create,
                       FileAccess.Write,
                       FileShare.None,
                       1 << 20))
            {
                for (var classIndex = 0; classIndex < classDirs.Length; classIndex++)
                {
                    foreach (var index in shuffledByClass[classIndex])
                    {
                        if (splitOf[index] != split)
                        {
                            continue;
                        }

                        WriteOut(stream, corpus, offsets[index], docSize[index]);
                        position += docSize[index];
                        newOffsets.Add(position);
                        newLabels.Add(classIndex);
                        perClass[split][classIndex]++;
                    }
                }
            }

            Npy.WriteInt64(Path.Combine(newDir, $"{Splits[split]}.offsets.npy"), newOffsets.ToArray());
            Npy.WriteInt64(Path.Combine(newDir, $"{Splits[split]}.labels.npy"), newLabels.ToArray());
            totals[split] = newLabels.Count;
        }

        var oldDir = Path.Combine(output, "packs.old");
        if (Directory.Exists(oldDir))
        {
            Directory.Delete(oldDir, true);
        }

        var packDir = Path.Combine(output, "packs");
        if (Directory.Exists(packDir))
        {
            Directory.Move(packDir, oldDir);
        }

        Directory.Move(newDir, packDir);
        if (Directory.Exists(oldDir))
        {
            Directory.Delete(oldDir, true);
        }

        // ---- dataset_info.json ----
        var info = new JsonObject
        {
            ["format"] = "thucnews-byte-pack",
            ["version"] = 1,
            ["source"] = source,
            ["tool"] = "tools/thucnews-dedup (C#)",
            ["seed"] = seed,
            ["ratios"] = new JsonArray(0.8, 0.1, 0.1),
            ["limit_per_class"] = limitPerClass,
            ["class_names"] = new JsonArray(classDirs.Select(dir => (JsonNode)dir.Name).ToArray()),
            ["splits"] = new JsonObject(),
            ["dedup"] = new JsonObject
            {
                ["enabled"] = dedup,
                ["method"] = $"simhash64(字符3-gram, 唯一 trigram 权重)，相似度 > 0.9（汉明距离 <= {MaxDistance}）",
                ["policy"] = "每个近重复簇保留编号最小的一篇",
                ["pairs"] = pairs,
                ["clusters"] = clusters,
                ["dropped"] = count - keep.Count(v => v),
                ["cross_class"] = new JsonObject
                {
                    ["clusters"] = crossClass.Clusters,
                    ["documents"] = crossClass.Documents,
                },
            },
        };
        for (var split = 0; split < Splits.Length; split++)
        {
            var classes = new JsonObject();
            for (var classIndex = 0; classIndex < classDirs.Length; classIndex++)
            {
                classes[classDirs[classIndex].Name] = perClass[split][classIndex];
            }

            info["splits"]![Splits[split]] = new JsonObject { ["total"] = totals[split], ["per_class"] = classes };
        }

        File.WriteAllText(
            Path.Combine(output, InfoName),
            info.ToJsonString(new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            }) + "\n",
            new UTF8Encoding(false));

        for (var split = 0; split < Splits.Length; split++)
        {
            Console.WriteLine($"  {Splits[split],-5}: {totals[split]} 篇");
        }

        Console.WriteLine($"完成: 共 {count} 篇 -> {keep.Count(v => v)} 篇，耗时 {timer.Elapsed.TotalSeconds:F1}s，输出 {output}");
        return 0;
    }

    private static (int Train, int Val, int Test) SplitCounts(int count)
    {
        var train = (int)(count * 0.8);
        var val = (int)(count * 0.1);
        var test = count - train - val;
        if (count >= 3)
        {
            if (val == 0)
            {
                train--;
                val = 1;
            }

            if (test == 0)
            {
                train--;
                test = 1;
            }
        }

        return (train, val, test);
    }

    private static ulong SeedFor(int seed, string className)
    {
        var hash = 14695981039346656037UL;
        foreach (var ch in $"{seed}:{className}")
        {
            hash ^= ch;
            hash *= 1099511628211UL;
        }

        return hash;
    }

    private const int ChunkShift = 30;
    private const long ChunkSize = 1L << ChunkShift;
    private const long ChunkMask = ChunkSize - 1;

    private static byte[][] CreateChunks(long totalBytes)
    {
        var count = (int)((totalBytes + ChunkSize - 1) >> ChunkShift);
        var chunks = new byte[count][];
        for (var i = 0; i < count; i++)
        {
            chunks[i] = new byte[(int)Math.Min(ChunkSize, totalBytes - (long)i * ChunkSize)];
        }

        return chunks;
    }

    private static void CopyInto(byte[][] chunks, long offset, ReadOnlySpan<byte> source)
    {
        while (!source.IsEmpty)
        {
            var chunk = chunks[(int)(offset >> ChunkShift)];
            var start = (int)(offset & ChunkMask);
            var take = Math.Min(chunk.Length - start, source.Length);
            source[..take].CopyTo(chunk.AsSpan(start, take));
            source = source[take..];
            offset += take;
        }
    }

    private static void ReadOut(byte[][] chunks, long offset, Span<byte> destination)
    {
        while (!destination.IsEmpty)
        {
            var chunk = chunks[(int)(offset >> ChunkShift)];
            var start = (int)(offset & ChunkMask);
            var take = Math.Min(chunk.Length - start, destination.Length);
            chunk.AsSpan(start, take).CopyTo(destination);
            destination = destination[take..];
            offset += take;
        }
    }

    private static string GetText(byte[][] chunks, long offset, int length)
    {
        var chunk = chunks[(int)(offset >> ChunkShift)];
        var start = (int)(offset & ChunkMask);
        if (start + length <= chunk.Length)
        {
            return Encoding.UTF8.GetString(chunk, start, length);
        }

        var buffer = new byte[length];
        ReadOut(chunks, offset, buffer);
        return Encoding.UTF8.GetString(buffer);
    }

    private static void WriteOut(Stream stream, byte[][] chunks, long offset, int length)
    {
        var chunk = chunks[(int)(offset >> ChunkShift)];
        var start = (int)(offset & ChunkMask);
        if (start + length <= chunk.Length)
        {
            stream.Write(chunk, start, length);
            return;
        }

        var buffer = new byte[length];
        ReadOut(chunks, offset, buffer);
        stream.Write(buffer, 0, length);
    }

    private static int CompareNumeric(FileInfo a, FileInfo b)
    {
        var left = int.TryParse(Path.GetFileNameWithoutExtension(a.Name), out var x) ? x : -1;
        var right = int.TryParse(Path.GetFileNameWithoutExtension(b.Name), out var y) ? y : -1;
        return left.CompareTo(right);
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
}
