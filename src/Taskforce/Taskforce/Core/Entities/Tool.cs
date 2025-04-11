using System;
using System.Threading.Tasks;
using System.Text.Json;

namespace Taskforce.Core.Entities
{
    public class Tool
    {
        public string Name { get; }
        public string Description { get; }
        public Type ArgumentType { get; }
        public Func<object, Task<object>> Execute { get; }

        public Tool(
            string name,
            string description,
            Type argumentType,
            Func<object, Task<object>> execute)
        {
            Name = name;
            Description = description;
            ArgumentType = argumentType;
            Execute = execute;
        }

        public ToolDefinition ToToolDefinition()
        {
            return new ToolDefinition
            {
                Name = Name,
                Description = Description,
                Parameters = GetParametersFromType(ArgumentType)
            };
        }

        private static Dictionary<string, ParameterInfo> GetParametersFromType(Type type)
        {
            var parameters = new Dictionary<string, ParameterInfo>();
            var properties = type.GetProperties();

            foreach (var prop in properties)
            {
                parameters[prop.Name.ToLower()] = new ParameterInfo
                {
                    Type = GetJsonType(prop.PropertyType),
                    Description = GetPropertyDescription(prop)
                };
            }

            return parameters;
        }

        private static string GetJsonType(Type type)
        {
            if (type == typeof(string)) return "string";
            if (type == typeof(int) || type == typeof(long)) return "integer";
            if (type == typeof(double) || type == typeof(float)) return "number";
            if (type == typeof(bool)) return "boolean";
            return "object";
        }

        private static string GetPropertyDescription(System.Reflection.PropertyInfo prop)
        {
            var descriptionAttr = prop.GetCustomAttributes(typeof(System.ComponentModel.DescriptionAttribute), true)
                .FirstOrDefault() as System.ComponentModel.DescriptionAttribute;
            return descriptionAttr?.Description ?? prop.Name;
        }
    }

    public class ToolCall
    {
        public string Name { get; set; }
        public object Arguments { get; set; }
        public string Id { get; set; }
    }

    public class ToolDefinition
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public Dictionary<string, ParameterInfo> Parameters { get; set; }
    }

    public class ParameterInfo
    {
        public string Type { get; set; }
        public string Description { get; set; }
    }
} 