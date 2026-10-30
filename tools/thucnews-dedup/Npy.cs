// npy（v1.0 / v2.0）的最小只读/写入实现：只支持 1 维 little-endian int64。
using System.Text;

internal static class Npy
{
    public static long[] ReadInt64(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        var magic = reader.ReadBytes(6);
        if (magic.Length != 6 || magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY")
        {
            throw new InvalidDataException($"不是 npy 文件: {path}");
        }

        var major = reader.ReadByte();
        reader.ReadByte();
        var headerLength = major >= 2 ? (int)reader.ReadUInt32() : reader.ReadUInt16();
        var header = Encoding.ASCII.GetString(reader.ReadBytes(headerLength));
        if (!header.Contains("'<i8'"))
        {
            throw new InvalidDataException($"只支持 '<i8'（int64）: {header}");
        }

        var count = ParseCount(header);
        var bytes = reader.ReadBytes(checked((int)(count * 8)));
        if (bytes.Length != count * 8)
        {
            throw new EndOfStreamException($"{path} 数据长度不对");
        }

        var values = new long[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    public static void WriteInt64(string path, long[] values)
    {
        var dict = $"{{'descr': '<i8', 'fortran_order': False, 'shape': ({values.Length},), }}";
        var unpadded = 10 + dict.Length + 1;
        var padding = (64 - unpadded % 64) % 64;
        var header = dict + new string(' ', padding) + "\n";

        using var writer = new BinaryWriter(File.Create(path));
        writer.Write((byte)0x93);
        writer.Write(Encoding.ASCII.GetBytes("NUMPY"));
        writer.Write((byte)1);
        writer.Write((byte)0);
        writer.Write((ushort)header.Length);
        writer.Write(Encoding.ASCII.GetBytes(header));

        var raw = new byte[values.Length * 8];
        Buffer.BlockCopy(values, 0, raw, 0, raw.Length);
        writer.Write(raw);
    }

    private static long ParseCount(string header)
    {
        var start = header.IndexOf('(');
        var end = header.IndexOf(',', start + 1);
        if (start < 0 || end < 0)
        {
            throw new InvalidDataException($"解析 shape 失败: {header}");
        }

        return long.Parse(header[(start + 1)..end].Trim());
    }
}
