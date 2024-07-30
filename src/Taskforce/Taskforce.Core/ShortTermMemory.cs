using Taskforce.Abstractions;

namespace Taskforce.Core
{
    /// <summary>
    /// Stores intermediate results during data processing. 
    /// For instance, when a something is being processed in stages, 
    /// short-term memory can hold the results of each stage before the final output is generated.
    /// </summary>
    public class ShortTermMemory : IMemory
    {
        private readonly List<string> _memory;

        public ShortTermMemory()
        {
            _memory = new List<string>();
        }

        public void Clear()
        {
            _memory.Clear();  
        }

        public string Get()
        {
            return _memory.Aggregate((a,b) => a + "\n" + b);
        }

        public void Store(string data)
        {
            Console.Out.WriteShortTermMemoryLine("ShortTermMemory gets updated...");

            _memory.Add(data);
        }
    }
}
