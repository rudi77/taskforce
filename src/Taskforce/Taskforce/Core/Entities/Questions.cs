using Newtonsoft.Json;

namespace Taskforce.Domain.Entities
{
    internal class Questions
    {
        [JsonProperty("sub-question")]
        public List<string> SubQuestions { get; set; }
    }
}
