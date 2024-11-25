namespace Taskforce.Domain.Interfaces
{
    /// <summary>
    /// Store and search for information in a memory component
    /// </summary>
    public interface IMemory
    {
        void Store(string data);

        string Get();

        void Clear();
    }
}