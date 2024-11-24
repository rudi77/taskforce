using Newtonsoft.Json;

namespace Taskforce.Core
{
    internal class Questions
    {
        [JsonProperty("sub-question")]
        public List<string> SubQuestions { get; set; }
    }
}
