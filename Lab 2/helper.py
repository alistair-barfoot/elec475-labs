import re

def parse_stats_to_latex(text):
    """
    Converts statistics text to LaTeX table format with & separators.
    
    Args:
        text (str): Multi-line string containing statistics
        
    Returns:
        str: Single line with values separated by ' & '
    """
    
    # Find all numbers (including decimals) in the text
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    
    # Join with ' & ' separator
    return ' & '.join(numbers)

# Example usage:
if __name__ == "__main__":
    input_text = """Results for vgg_BOTH_AUG.pth:
  Overall - Min:   0, Max: 110, Mean:   9, Stdev: 11.1
  Worst 4 - Min:  84, Max: 110, Mean:  94, Stdev: 9.8
  Best 4  - Min:   0, Max:   1, Mean:   0, Stdev: 0.1"""
    
    result = parse_stats_to_latex(input_text) + "\\\\"
    print(result)