namespace Taskforce.Infrastructure.Observability
{
    public static class Extensions
    {
        private static readonly object _consoleLock = new object();
        private static readonly SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);

        public static async Task WriteAgentLineAsync(this TextWriter writer, string content)
        {
            await Console.Out.WriteColorConsoleAsync(content, ConsoleColor.Green);
        }

        public static async Task WriteLmmLineAsync(this TextWriter writer, string content)
        {
            await Console.Out.WriteColorConsoleAsync(content, ConsoleColor.Cyan);
        }

        public static async Task WriteShortTermMemoryLineAsync(this TextWriter writer, string content)
        {
            await Console.Out.WriteColorConsoleAsync(content, ConsoleColor.Magenta);
        }

        public static async Task WritePlannerLineAsync(this TextWriter writer, string content)
        {
            await Console.Out.WriteColorConsoleAsync(content, ConsoleColor.Yellow);
        }

        public static void WriteAgentLine(this TextWriter writer, string content)
        {
            Console.Out.WriteColorConsole(content, ConsoleColor.Green);
        }

        public static void WriteLmmLine(this TextWriter writer, string content)
        {
            Console.Out.WriteColorConsole(content, ConsoleColor.Cyan);
        }

        public static void WriteShortTermMemoryLine(this TextWriter writer, string content)
        {
            Console.Out.WriteColorConsole(content, ConsoleColor.Magenta);
        }

        public static void WritePlannerLine(this TextWriter writer, string content)
        {
            Console.Out.WriteColorConsole(content, ConsoleColor.Yellow);
        }

        public static async Task WriteColorConsoleAsync(this TextWriter writer, string content, ConsoleColor foreground)
        {
            await _semaphore.WaitAsync();
            try
            {
                Console.ForegroundColor = foreground;
                await writer.WriteLineAsync(content);
                Console.ResetColor();
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public static void WriteColorConsole(this TextWriter writer, string content, ConsoleColor foreground)
        {
            lock (_consoleLock)
            {
                Console.ForegroundColor = foreground;
                writer.WriteLine(content);
                Console.ResetColor();
            }
        }
    }
}