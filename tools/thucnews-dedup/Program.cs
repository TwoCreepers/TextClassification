// 全量 SimHash 近重复检测（异步预读流水线：读并发 / 计算并发分开）。
// 用法:
//   bench <manifest> <read并发> <read|full> [计算并发]      基准
//   dedup <manifest> <read并发> <pairs.txt> [计算并发]      全量近重复检测

using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Threading.Channels;

internal static class Program
{
    private const int MaxDistance = 6; // SimHash 相似度 1 - d/64 > 0.9

    internal sealed class LocalState
    {
        public readonly char[] Buffer = new char[1 << 21];
        public readonly int[] Acc = new int[64];
        public readonly HashSet<ulong> Seen = new(8192);
    }

    internal static ulong SimHash(string text, LocalState state)
    {
        var buffer = state.Buffer;
        var acc = state.Acc;
        var seen = state.Seen;
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
                    hashes[index] = SimHash(Encoding.UTF8.GetString(data), state);
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

        const int windowBits = 16;
        const int stride = 8;
        var keys = new ulong[hashes.Length];
        var sorted = new int[hashes.Length];
        var counts = new int[1 << windowBits];
        var pairs = new HashSet<long>();
        var distances = new int[MaxDistance + 1];
        var pairCount = 0;

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
                        var distance = BitOperations.PopCount(hashes[i] ^ hashes[j]);
                        if (distance > MaxDistance)
                        {
                            continue;
                        }

                        var lo = Math.Min(i, j);
                        var hi = Math.Max(i, j);
                        if (pairs.Add(((long)lo << 32) | (uint)hi))
                        {
                            distances[distance]++;
                            pairCount++;
                        }
                    }
                }

                start = end;
            }

            Console.WriteLine($"  窗口 {offset,2}-{offset + windowBits - 1,2}: 候选对累计 {pairCount}");
        }

        stopwatch.Stop();
        Console.WriteLine($"近重复对（汉明距离 <= {MaxDistance}）: {pairCount}");
        for (var d = 0; d <= MaxDistance; d++)
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
                hashes[i] = SimHash(Encoding.UTF8.GetString(File.ReadAllBytes(files[i])), state);
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
            "dedup-pack" => Task.FromResult(PackDedup.Run(args[1], int.Parse(args[2]), !args.Contains("--keep-cross-class"))),
            "prep" => Task.FromResult(RunPrep(args)),
            _ => throw new ArgumentException($"unknown command: {command}"),
        };
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
