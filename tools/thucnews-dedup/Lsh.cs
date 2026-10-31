// SimHash 近重复候选生成：7 个 16-bit 重叠窗口分桶，桶内配对 + 1-bit 邻桶配对。
//
// 为什么这样能保证不漏：一对 hash 若有 d 个差异位，每个差异位最多落进 2 个窗口；
// 要让 7 个窗口每个都至少含 1 个差异位，需要 >= 14 次「窗口 × 差异位」命中，
// 而 d <= 6 最多只能提供 12 次，所以必然存在一个窗口最多含 1 个差异位：
//   0 个 -> 两个 hash 同桶，桶内配对命中；
//   1 个 -> 翻转那一位后同桶，1-bit 邻桶配对命中。
// 旧实现只做了同桶配对，等价于只保证 d <= 3；把窗口错开/加密并不能拿到保证（ml/README.md 里有反例），
// 必须补 1-bit 邻桶。反过来，MaxDistance 也不能再往上调：d = 7 时 7 个窗口可能每个都含 2 个差异位。

using System.Numerics;

internal static class Lsh
{
    public const int MaxDistance = 6;

    private const int WindowBits = 16;
    private const int Stride = 8;
    private const int WindowSize = 1 << WindowBits;

    internal readonly record struct Stats(long Candidates, long IdenticalPairs);

    private sealed class PairComparer : IEqualityComparer<long>
    {
        public bool Equals(long x, long y) => x == y;

        // (lo << 32 | hi) 直接用默认的 long.GetHashCode 会退化成 lo ^ hi，
        // 同一个近重复簇的对几乎全撞进 ~2m 个桶，链长 O(m)；这里按 splitmix 打散。
        public int GetHashCode(long value)
        {
            var z = (ulong)value;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return (int)(z ^ (z >> 31));
        }
    }

    /// <summary>
    /// 把 <paramref name="hashes"/> 并成近重复簇：先按完全相同的 hash 合并（O(n)，避免 m 篇同稿产生 C(m,2) 对），
    /// 再对去重后的 hash 跑多探针 LSH，覆盖所有汉明距离 &lt;= <see cref="MaxDistance"/> 的对。
    /// </summary>
    /// <returns>并查集数组，可直接交给 <see cref="UnionFind.Find"/> 或 <c>PackDedup.AnalyzeCrossClass</c>。</returns>
    public static int[] BuildClusters(ulong[] hashes, out Stats stats)
    {
        var parent = UnionFind.Create(hashes.Length);

        var first = new Dictionary<ulong, int>(hashes.Length);
        var groupSize = new int[hashes.Length];
        var distinct = new ulong[hashes.Length];
        var origin = new int[hashes.Length];
        var distinctCount = 0;
        var identicalPairs = 0L;

        for (var index = 0; index < hashes.Length; index++)
        {
            var hash = hashes[index];
            if (first.TryGetValue(hash, out var representative))
            {
                identicalPairs += groupSize[representative];
                groupSize[representative]++;
                UnionFind.Union(parent, representative, index);
                continue;
            }

            first.Add(hash, index);
            groupSize[index] = 1;
            distinct[distinctCount] = hash;
            origin[distinctCount] = index;
            distinctCount++;
        }

        Array.Resize(ref distinct, distinctCount);
        Array.Resize(ref origin, distinctCount);
        var candidates = Scan(distinct, origin, parent, null);
        stats = new Stats(candidates, identicalPairs);
        return parent;
    }

    /// <summary>返回所有汉明距离 &lt;= MaxDistance 的文档对（编码为 <c>(long)lo &lt;&lt; 32 | (uint)hi</c>），用于诊断输出。</summary>
    public static HashSet<long> BuildPairs(ulong[] hashes)
    {
        var origin = new int[hashes.Length];
        var parent = UnionFind.Create(hashes.Length);
        var pairs = new HashSet<long>(new PairComparer());
        for (var i = 0; i < origin.Length; i++)
        {
            origin[i] = i;
        }

        Scan(
            hashes,
            origin,
            parent,
            (a, b) =>
            {
                var lo = Math.Min(a, b);
                var hi = Math.Max(a, b);
                pairs.Add(((long)lo << 32) | (uint)hi);
            });
        return pairs;
    }

    private static long Scan(ulong[] values, int[] origin, int[] parent, Action<int, int>? onPair)
    {
        var count = values.Length;
        if (count < 2)
        {
            return 0;
        }

        var keys = new int[count];
        var sorted = new int[count];
        var counts = new int[WindowSize];
        var candidates = 0L;

        for (var offset = 0; offset + WindowBits <= 64; offset += Stride)
        {
            Array.Clear(counts);
            for (var i = 0; i < count; i++)
            {
                var key = (int)((values[i] >> offset) & (WindowSize - 1));
                keys[i] = key;
                counts[key]++;
            }

            var sum = 0;
            for (var k = 0; k < WindowSize; k++)
            {
                var bucketSize = counts[k];
                counts[k] = sum;
                sum += bucketSize;
            }

            for (var i = 0; i < count; i++)
            {
                sorted[counts[keys[i]]++] = i;
            }

            // 计数排序后 counts[k] 是桶 k 的桶尾；桶 k 的桶头就是上一个桶的桶尾。
            var start = 0;
            for (var k = 0; k < WindowSize; k++)
            {
                var end = counts[k];
                if (start != end)
                {
                    for (var a = start; a < end; a++)
                    {
                        for (var b = a + 1; b < end; b++)
                        {
                            candidates++;
                            Consider(values, origin, parent, onPair, sorted[a], sorted[b]);
                        }
                    }

                    // 1-bit 邻桶：只从较小的桶号处理一次，保证每个无序桶对只扫一遍。
                    for (var bit = 0; bit < WindowBits; bit++)
                    {
                        var neighbor = k ^ (1 << bit);
                        if (neighbor < k)
                        {
                            continue;
                        }

                        var neighborStart = counts[neighbor - 1];
                        var neighborEnd = counts[neighbor];
                        for (var a = start; a < end; a++)
                        {
                            var i = sorted[a];
                            for (var b = neighborStart; b < neighborEnd; b++)
                            {
                                candidates++;
                                Consider(values, origin, parent, onPair, i, sorted[b]);
                            }
                        }
                    }
                }

                start = end;
            }
        }

        return candidates;
    }

    private static void Consider(
        ulong[] values,
        int[] origin,
        int[] parent,
        Action<int, int>? onPair,
        int i,
        int j)
    {
        if (BitOperations.PopCount(values[i] ^ values[j]) > MaxDistance)
        {
            return;
        }

        var a = origin[i];
        var b = origin[j];
        UnionFind.Union(parent, a, b);
        onPair?.Invoke(a, b);
    }
}
