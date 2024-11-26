using Taskforce.Domain.Interfaces;

namespace Taskforce.Core.Entities
{
    public class MemoryManager
    {
        private readonly IMemory _shortTermMemory;

        public MemoryManager(IMemory shortTermMemory)
        {
            _shortTermMemory = shortTermMemory;
        }

        public void Store(string data)
        {
            _shortTermMemory.Store(data);
        }

        public string Retrieve()
        {
            return _shortTermMemory.Get();
        }

        public void Clear()
        {
            _shortTermMemory.Clear();
        }
    }

}
