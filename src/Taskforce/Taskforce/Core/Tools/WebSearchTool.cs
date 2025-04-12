using System.ComponentModel;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using DuckDuckGo.Net;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Taskforce.Core.Entities;

namespace Taskforce.Core.Tools
{
    /// <summary>
    /// Tool for performing web searches and returning relevant results.
    /// Uses DuckDuckGo's API which is free and doesn't require an API key.
    /// </summary>
    public class WebSearchTool : Tool
    {
        private readonly ILogger _logger;
        private int _maxResults;

        public WebSearchTool(ILogger? logger = null, int maxResults = 5) 
            : base(
                name: "web_search",
                description: "Search the web using DuckDuckGo API",
                argumentType: typeof(WebSearchArguments),
                execute: ExecuteSearch)
        {
            _logger = logger ?? NullLogger.Instance;
            _maxResults = maxResults;
        }

        private static async Task<object> ExecuteSearch(object args)
        {
            if (args is string jsonArgs)
            {
                try
                {
                    var searchArgs = JsonSerializer.Deserialize<WebSearchArguments>(jsonArgs);
                    if (searchArgs == null)
                    {
                        throw new ArgumentException("Failed to deserialize search arguments");
                    }
                    return await PerformWebSearch(searchArgs);
                }
                catch (JsonException ex)
                {
                    throw new ArgumentException($"Invalid search arguments format: {ex.Message}");
                }
            }
            else if (args is WebSearchArguments searchArgs)
            {
                return await PerformWebSearch(searchArgs);
            }
            else
            {
                throw new ArgumentException($"Arguments must be of type WebSearchArguments or a JSON string representing WebSearchArguments");
            }
        }

        private static async Task<string> PerformWebSearch(WebSearchArguments args)
        {
            var search = new Search();
            var searchResult = search.Query(args.Query, "Taskforce");

            return FormatSearchResults(searchResult);
        }

        private static string FormatSearchResults(DuckDuckGo.Net.SearchResult? response)
        {
            if (response == null || response.Results == null || !response.Results.Any())
            {
                return "No search results found.";
            }

            var sb = new StringBuilder();
            foreach (var item in response.Results.Take(5))
            {
                sb.AppendLine($"Link: {item.FirstUrl}");
                sb.AppendLine($"Text: {item.Text}");
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }

    /// <summary>
    /// Arguments for web search operations
    /// </summary>
    public class WebSearchArguments
    {
        /// <summary>
        /// The search query to execute
        /// </summary>
        [Description("The search query to execute")]
        [JsonPropertyName("query")]
        public string Query { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents a single search result
    /// </summary>
    public class SearchResult
    {
        public string Title { get; set; } = string.Empty;
        public string Link { get; set; } = string.Empty;
        public string Snippet { get; set; } = string.Empty;
    }

    /// <summary>
    /// Response structure for DuckDuckGo API
    /// </summary>
    public class GoogleSearchResponse
    {
        public List<SearchResult> Items { get; set; } = new();
    }
} 