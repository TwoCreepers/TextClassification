// SimHash 近重复去重的命令入口：prep（去重 + 划分）、dedup-pack（对已有 pack 补去重）、
// dedup（诊断：把近重复对写进 pairs.txt）、bench / sync / purge、selftest（暴力枚举校验 LSH）。
// 不带参数运行会打印用法。

using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Threading.Channels;

internal static class Program
{
    internal sealed class LocalState
    {
        public readonly char[] Buffer = new char[1 << 21];
        public readonly int[] Acc = new int[64];
        public readonly HashSet<ulong> Seen = new(8192);
    }

    internal static ulong SimHash(string text, LocalState state) => SimHash(text.AsSpan(), state);

    /// <summary>直接对 UTF-8 字节做 SimHash，省掉每篇一个 string 的分配。</summary>
    internal static ulong SimHash(ReadOnlySpan<byte> utf8, LocalState state)
    {
        if (utf8.Length > state.Buffer.Length)
        {
            throw new InvalidDataException($"单篇文档 {utf8.Length} 字节超出 SimHash 缓冲区 {state.Buffer.Length}");
        }

        var length = Encoding.UTF8.GetChars(utf8, state.Buffer);
        return SimHash(state.Buffer.AsSpan(0, length), state);
    }

    internal static ulong SimHash(ReadOnlySpan<char> text, LocalState state)
    {
        var buffer = state.Buffer;
        var acc = state.Acc;
        var seen = state.Seen;
        if (text.Length > buffer.Length)
        {
            throw new InvalidDataException($"单篇文档 {text.Length} 字符超出 SimHash 缓冲区 {buffer.Length}");
        }

        var n = 0;
        foreach (var c in text)
        {
            if (c <= 0x20 || c == 0x3000 || c == 0xA0 || c == 0xFEFF)
            {
                continue;
            }

            buffer[n++] = c;
        }

        Array.Clear(acc);
        seen.Clear();
        for (var i = 0; i + 2 < n; i++)
        {
            var key = ((ulong)buffer[i] << 32) | ((ulong)buffer[i + 1] << 16) | buffer[i + 2];
            if (!seen.Add(key))
            {
                continue;
            }

            var h = 0x9E3779B97F4A7C15UL ^ key;
            h ^= h >> 30;
            h *= 0xBF58476D1CE4E5B9UL;
            h ^= h >> 27;
            h *= 0x94D049BB133111EBUL;
            h ^= h >> 31;
            for (var b = 0; b < 64; b++)
            {
                acc[b] += (int)((h >> b) & 1) * 2 - 1;
            }
        }

        var value = 0UL;
        for (var b = 0; b < 64; b++)
        {
            if (acc[b] > 0)
            {
                value |= 1UL << b;
            }
        }

        return value;
    }

    private static async Task<long> ReadAllAsync(string[] files, int readConcurrency)
    {
        var total = 0L;
        await Parallel.ForEachAsync(
            Enumerable.Range(0, files.Length),
            new ParallelOptions { MaxDegreeOfParallelism = readConcurrency },
            async (index, token) =>
            {
                var bytes = await File.ReadAllBytesAsync(files[index], token).ConfigureAwait(false);
                Interlocked.Add(ref total, bytes.Length);
            }).ConfigureAwait(false);
        return total;
    }

    /// <summary>读取并发与计算并发分离的流水线：读线程把文件读进有界 Channel，计算线程边算边取。</summary>
    private static async Task<ulong[]> HashAllAsync(string[] files, int readConcurrency, int computeConcurrency)
    {
        var hashes = new ulong[files.Length];
        var channel = Channel.CreateBounded<(int Index, byte[] Data)>(
            new BoundedChannelOptions(512)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleWriter = false,
                SingleReader = false,
            });

        var producer = Task.Run(async () =>
        {
            await Parallel.ForEachAsync(
                Enumerable.Range(0, files.Length),
                new ParallelOptions { MaxDegreeOfParallelism = readConcurrency },
                async (index, token) =>
                {
                    var bytes = await File.ReadAllBytesAsync(files[index], token).ConfigureAwait(false);
                    await channel.Writer.WriteAsync((index, bytes), token).ConfigureAwait(false);
                }).ConfigureAwait(false);
            channel.Writer.Complete();
        });

        var consumers = new Task[computeConcurrency];
        for (var worker = 0; worker < computeConcurrency; worker++)
        {
            consumers[worker] = Task.Run(async () =>
            {
                var state = new LocalState();
                await foreach (var (index, data) in channel.Reader.ReadAllAsync().ConfigureAwait(false))
                {
                    hashes[index] = SimHash(data, state);
                }
            });
        }

        await Task.WhenAll(consumers).ConfigureAwait(false);
        await producer.ConfigureAwait(false);
        return hashes;
    }

    private static async Task<int> RunDedupAsync(string manifest, int readConcurrency, string output, int computeConcurrency)
    {
        var files = File.ReadAllLines(manifest);
        var stopwatch = Stopwatch.StartNew();
        var hashes = await HashAllAsync(files, readConcurrency, computeConcurrency).ConfigureAwait(false);
        Console.WriteLine($"哈希完成: {files.Length} 篇, {stopwatch.Elapsed.TotalSeconds:F1}s");

        var pairs = Lsh.BuildPairs(hashes);
        var distances = new int[Lsh.MaxDistance + 1];
        foreach (var pair in pairs)
        {
            var lo = (int)(pair >> 32);
            var hi = (int)(pair & 0xFFFFFFFF);
            distances[BitOperations.PopCount(hashes[lo] ^ hashes[hi])]++;
        }

        stopwatch.Stop();
        Console.WriteLine($"近重复对（汉明距离 <= {Lsh.MaxDistance}）: {pairs.Count}");
        for (var d = 0; d <= Lsh.MaxDistance; d++)
        {
            Console.WriteLine($"  距离 {d}: {distances[d]} 对");
        }

        await using (var writer = new StreamWriter(output))
        {
            foreach (var pair in pairs)
            {
                var lo = (int)(pair >> 32);
                var hi = (int)(pair & 0xFFFFFFFF);
                await writer.WriteLineAsync($"{lo}\t{hi}\t{BitOperations.PopCount(hashes[lo] ^ hashes[hi])}").ConfigureAwait(false);
            }
        }

        Console.WriteLine($"总耗时 {stopwatch.Elapsed.TotalSeconds:F1}s，结果写入 {output}");
        return 0;
    }

    private static async Task<int> RunBenchAsync(string manifest, int readConcurrency, string mode, int computeConcurrency)
    {
        var files = File.ReadAllLines(manifest);
        var stopwatch = Stopwatch.StartNew();
        long hi = 0;
        long lo = 0;

        if (mode == "read")
        {
            await ReadAllAsync(files, readConcurrency).ConfigureAwait(false);
        }
        else
        {
            var hashes = await HashAllAsync(files, readConcurrency, computeConcurrency).ConfigureAwait(false);
            foreach (var value in hashes)
            {
                hi += (long)(value >> 32);
                lo += (long)(value & 0xFFFFFFFF);
            }
        }

        var elapsed = stopwatch.Elapsed.TotalSeconds;
        Console.WriteLine(
            $"lang=csharp mode={mode} files={files.Length} read={readConcurrency} compute={computeConcurrency} " +
            $"seconds={elapsed:F3} docs_per_sec={files.Length / elapsed:F1} checksum={(hi & 0xFFFFFFFF):x8}-{(lo & 0xFFFFFFFF):x8}");
        return 0;
    }

    /// <summary>对照组：同步阻塞读 + 就地计算（优化前的写法）。</summary>
    private static int RunSyncBench(string manifest, int threads)
    {
        var files = File.ReadAllLines(manifest);
        var hashes = new ulong[files.Length];
        var stopwatch = Stopwatch.StartNew();
        Parallel.For(
            0,
            files.Length,
            new ParallelOptions { MaxDegreeOfParallelism = threads },
            () => new LocalState(),
            (i, _, state) =>
            {
                hashes[i] = SimHash(File.ReadAllBytes(files[i]), state);
                return state;
            },
            _ => { });
        long hi = 0;
        long lo = 0;
        foreach (var value in hashes)
        {
            hi += (long)(value >> 32);
            lo += (long)(value & 0xFFFFFFFF);
        }

        var elapsed = stopwatch.Elapsed.TotalSeconds;
        Console.WriteLine(
            $"lang=csharp mode=sync files={files.Length} read={threads} compute={threads} " +
            $"seconds={elapsed:F3} docs_per_sec={files.Length / elapsed:F1} checksum={(hi & 0xFFFFFFFF):x8}-{(lo & 0xFFFFFFFF):x8}");
        return 0;
    }

    /// <summary>非管理员下的“清缓存”手段：吃满大块内存，逼 Windows 回收 standby 里的文件页。</summary>
    private static int RunPurge(int gigabytes)
    {
        var stopwatch = Stopwatch.StartNew();
        var chunks = new List<byte[]>(gigabytes);
        for (var i = 0; i < gigabytes; i++)
        {
            var chunk = GC.AllocateUninitializedArray<byte>(1 << 30);
            for (var offset = 0; offset < chunk.Length; offset += 4096)
            {
                chunk[offset] = 1;
            }

            chunks.Add(chunk);
        }

        Console.WriteLine($"purge: 占用 {gigabytes} GiB 并逐页写入，用时 {stopwatch.Elapsed.TotalSeconds:F1}s");
        chunks.Clear();
        GC.Collect();
        return 0;
    }

    private static Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine(
                "用法:\n"
                + "  dedup-pack <root> <并发> [--keep-cross-class] [--dry-run]   对已有 pack 补一次去重（幂等）\n"
                + "  prep <source> <output> [--seed N] [--limit-per-class N] [--workers N] [--no-dedup] [--keep-cross-class] [--overwrite]\n"
                + "  dedup <manifest> <读并发> <pairs.txt> [计算并发]   全量近重复检测（诊断用）\n"
                + "  bench <manifest> <读并发> <read|full> [计算并发]\n"
                + "  sync <manifest> <并发>\n"
                + "  purge <GiB>\n"
                + "  selftest                                          用暴力枚举校验 LSH 的完整性");
            return Task.FromResult(1);
        }

        var command = args[0];
        return command switch
        {
            "dedup" => RunDedupAsync(
                args[1],
                int.Parse(args[2]),
                args[3],
                args.Length > 4 ? int.Parse(args[4]) : Environment.ProcessorCount),
            "bench" => RunBenchAsync(
                args[1],
                int.Parse(args[2]),
                args[3],
                args.Length > 4 ? int.Parse(args[4]) : Environment.ProcessorCount),
            "sync" => Task.FromResult(RunSyncBench(args[1], int.Parse(args[2]))),
            "purge" => Task.FromResult(RunPurge(int.Parse(args[1]))),
            "dedup-pack" => Task.FromResult(
                PackDedup.Run(
                    args[1],
                    int.Parse(args[2]),
                    !args.Contains("--keep-cross-class"),
                    args.Contains("--dry-run"))),
            "prep" => Task.FromResult(RunPrep(args)),
            "selftest" => Task.FromResult(RunSelfTest()),
            _ => throw new ArgumentException($"unknown command: {command}"),
        };
    }

    /// <summary>用暴力枚举校验 LSH：随机用例 + 对抗性差异位模式，连通分量必须与 O(n²) 全比对完全一致。</summary>
    private static int RunSelfTest()
    {
        var random = new Random(20261031);
        var cases = 0;
        var failures = 0;

        // 这些差异位模式正是 7 个 16-bit 同桶窗口会整对漏掉的（旧实现 0 对的那些）。
        int[][] adversarial =
        {
            new[] { 8, 24, 40, 48 },
            new[] { 8, 16, 24, 32, 40, 48 },
            new[] { 0, 12, 24, 36, 48, 60 },
            new[] { 1, 9, 17, 25, 33, 41, 49, 57 },
        };
        foreach (var bits in adversarial)
        {
            var a = NextUInt64(random);
            var b = a;
            foreach (var bit in bits)
            {
                b ^= 1UL << bit;
            }

            cases++;
            failures += CompareClustersWithBruteForce(new[] { a, b });
        }

        for (var trial = 0; trial < 300; trial++)
        {
            var n = random.Next(2, 400);
            var hashes = new ulong[n];
            for (var i = 0; i < n; i++)
            {
                var roll = random.Next(100);
                if (i == 0 || roll < 45)
                {
                    hashes[i] = NextUInt64(random);
                }
                else if (roll < 60)
                {
                    hashes[i] = hashes[random.Next(i)]; // 完全重复
                }
                else
                {
                    hashes[i] = hashes[random.Next(i)] ^ FlipBits(random, random.Next(1, Lsh.MaxDistance + 1)); // 近重复
                }
            }

            cases++;
            failures += CompareClustersWithBruteForce(hashes);

            if (trial < 30)
            {
                cases++;
                failures += ComparePairsWithBruteForce(hashes);
            }
        }

        var duplicates = new ulong[500];
        Array.Fill(duplicates, NextUInt64(random));
        cases++;
        failures += CompareClustersWithBruteForce(duplicates);

        Console.WriteLine(failures == 0
            ? $"selftest: OK（{cases} 组用例，与暴力枚举完全一致）"
            : $"selftest: FAILED（{cases} 组用例，{failures} 处与暴力枚举不一致）");
        return failures == 0 ? 0 : 1;
    }

    private static int CompareClustersWithBruteForce(ulong[] hashes)
    {
        var expected = BuildBruteForceClusters(hashes);
        var actual = Lsh.BuildClusters(hashes, out _);
        var mismatches = 0;
        for (var i = 0; i < hashes.Length; i++)
        {
            for (var j = i + 1; j < hashes.Length; j++)
            {
                var expectedSame = UnionFind.Find(expected, i) == UnionFind.Find(expected, j);
                var actualSame = UnionFind.Find(actual, i) == UnionFind.Find(actual, j);
                if (expectedSame != actualSame)
                {
                    mismatches++;
                }
            }
        }

        return mismatches;
    }

    private static int ComparePairsWithBruteForce(ulong[] hashes)
    {
        var expected = 0;
        for (var i = 0; i < hashes.Length; i++)
        {
            for (var j = i + 1; j < hashes.Length; j++)
            {
                if (BitOperations.PopCount(hashes[i] ^ hashes[j]) <= Lsh.MaxDistance)
                {
                    expected++;
                }
            }
        }

        var actual = Lsh.BuildPairs(hashes).Count;
        return expected == actual ? 0 : 1;
    }

    private static int[] BuildBruteForceClusters(ulong[] hashes)
    {
        var parent = UnionFind.Create(hashes.Length);
        for (var i = 0; i < hashes.Length; i++)
        {
            for (var j = i + 1; j < hashes.Length; j++)
            {
                if (BitOperations.PopCount(hashes[i] ^ hashes[j]) <= Lsh.MaxDistance)
                {
                    UnionFind.Union(parent, i, j);
                }
            }
        }

        return parent;
    }

    private static ulong NextUInt64(Random random) => ((ulong)(uint)random.Next() << 32) | (uint)random.Next();

    private static ulong FlipBits(Random random, int count)
    {
        var mask = 0UL;
        var flipped = 0;
        while (flipped < count)
        {
            var bit = random.Next(64);
            if (((mask >> bit) & 1) != 0)
            {
                continue;
            }

            mask |= 1UL << bit;
            flipped++;
        }

        return mask;
    }

    /// <summary>prep &lt;source&gt; &lt;output&gt; [--seed N] [--limit-per-class N] [--workers N] [--no-dedup] [--keep-cross-class] [--overwrite]</summary>
    private static int RunPrep(string[] args)
    {
        var seed = 42;
        var limitPerClass = 0;
        var workers = Environment.ProcessorCount;
        var dedup = true;
        var dropCrossClass = true;
        var overwrite = false;
        for (var i = 3; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--seed":
                    seed = int.Parse(args[++i]);
                    break;
                case "--limit-per-class":
                    limitPerClass = int.Parse(args[++i]);
                    break;
                case "--workers":
                    workers = int.Parse(args[++i]);
                    break;
                case "--no-dedup":
                    dedup = false;
                    break;
                case "--keep-cross-class":
                    dropCrossClass = false;
                    break;
                case "--overwrite":
                    overwrite = true;
                    break;
                default:
                    throw new ArgumentException($"未知参数: {args[i]}");
            }
        }

        return Prep.Run(args[1], args[2], seed, limitPerClass, workers, dedup, overwrite, dropCrossClass);
    }
}
