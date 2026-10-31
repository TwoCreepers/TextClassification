// 并查集：去重和跨类统计共用，避免各调用方各写一份 Find/Union。
internal static class UnionFind
{
    public static int[] Create(int count)
    {
        var parent = new int[count];
        for (var i = 0; i < count; i++)
        {
            parent[i] = i;
        }

        return parent;
    }

    public static int Find(int[] parent, int index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }

        return index;
    }

    /// <summary>合并两个集合，返回是否真的发生了合并。</summary>
    public static bool Union(int[] parent, int a, int b)
    {
        var rootA = Find(parent, a);
        var rootB = Find(parent, b);
        if (rootA == rootB)
        {
            return false;
        }

        parent[rootB] = rootA;
        return true;
    }
}
