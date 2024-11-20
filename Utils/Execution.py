import ast
import base64
import importlib
import io
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Union
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

@dataclass
class ExecutionResult:
    """Class to store the results of code execution"""
    status: bool
    output: Any
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    generated_figures: List[str] = None
    printed_output: str = ""
    dataframes: Dict[str, pd.DataFrame] = None

class CodeIndenter:
    """Helper class to handle code indentation"""
    
    @staticmethod
    def auto_indent(code: str, indent_size: int = 4) -> str:
        """
        Automatically indent the given code
        
        Args:
            code (str): The code to indent
            indent_size (int): Number of spaces for indentation
        
        Returns:
            str: Properly indented code
        """
        lines = code.split('\n')
        
        lines = [line.strip() for line in lines]
        
        lines = [line for line in lines if line]
        
        if any(line.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')) for line in lines):
            indented_lines = []
            current_indent = 0
            
            for line in lines:
                # Decrease indent for lines starting with closing blocks
                if line.startswith(('return', 'break', 'continue', 'pass', 'except', 'elif', 'else:', 'finally:')):
                    current_indent = max(0, current_indent - 1)
                
                # Add indentation
                indented_line = ' ' * (current_indent * indent_size) + line
                indented_lines.append(indented_line)
                
                # Increase indent for lines that typically start a new block
                if (line.endswith(':') or 
                    line.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with '))):
                    current_indent += 1
                
                # Decrease indent after certain keywords
                if line.startswith(('return', 'break', 'continue', 'pass')):
                    current_indent = max(0, current_indent - 1)
            
            return '\n'.join(indented_lines)
        
        return '\n'.join(lines)  # Return original lines if no special indentation needed

class CodeExecutor(CodeIndenter):
    """Enhanced code executor with automatic indentation"""
    
    def __init__(self, capture_prints: bool = True, max_output_length: int = 10000):
        super().__init__()  # Initialize CodeIndenter methods
        self.capture_prints = capture_prints
        self.max_output_length = max_output_length
        self.default_imports = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'display': display,
            'HTML': HTML
        }

    def _preprocess_code(self, code: str) -> str:
        """Clean, preprocess, and auto-indent the input code"""
        # Remove markdown code blocks if present
        if "```" in code:
            pattern = r"```(?:python|py)?\n([\s\S]+?)```"
            matches = re.findall(pattern, code)
            if matches:
                code = matches[0]
            code = code.replace("```", "")
        
        # Remove common explanation text markers
        markers = ["<imports>", "<stub>", "<transforms>", "<code>", "</code>"]
        for marker in markers:
            code = code.replace(marker, "")
        
        # Auto-indent the code
        code = self.auto_indent(code)
        
        return code.strip()

    def _get_imports(self, code_string: str) -> Dict[str, Any]:
        """Extract and process imports from the code"""
        globals_dict = self.default_imports.copy()
        
        try:
            tree = ast.parse(code_string)
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            module = importlib.import_module(alias.name)
                            globals_dict[alias.asname or alias.name] = module
                        except ImportError:
                            pass
                elif isinstance(node, ast.ImportFrom):
                    try:
                        module = importlib.import_module(node.module)
                        for alias in node.names:
                            obj = getattr(module, alias.name)
                            globals_dict[alias.asname or alias.name] = obj
                    except ImportError:
                        pass
        except SyntaxError:
            pass  # If there's a syntax error, we'll catch it during execution
            
        return globals_dict

    def _capture_output(self):
        """Create string buffer to capture printed output"""
        self.string_buffer = io.StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.string_buffer

    def _restore_output(self) -> str:
        """Restore original stdout and return captured output"""
        sys.stdout = self.original_stdout
        output = self.string_buffer.getvalue()
        self.string_buffer.close()
        return output

    def _capture_figures(self) -> List[str]:
        """Capture all currently open matplotlib figures as base64 strings"""
        figures = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            figures.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            buf.close()
        plt.close('all')
        return figures

    def _find_dataframes(self, locals_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Find all pandas DataFrames in the locals dictionary"""
        return {
            name: obj for name, obj in locals_dict.items()
            if isinstance(obj, pd.DataFrame)
        }

    def execute(self, code: str, data: Any = None) -> ExecutionResult:
        """
        Execute the provided code and return the results
        
        Args:
            code (str): The Python code to execute
            data (Any, optional): Data to make available in the execution context
            
        Returns:
            ExecutionResult: Object containing execution results and any outputs
        """
        code = self._preprocess_code(code)
        globals_dict = self._get_imports(code)
        
        if data is not None:
            globals_dict['data'] = data
            
        locals_dict = {}
        
        try:
            if self.capture_prints:
                self._capture_output()
                
            # Execute the code
            exec(code, globals_dict, locals_dict)
            
            printed_output = self._restore_output() if self.capture_prints else ""
            if len(printed_output) > self.max_output_length:
                printed_output = printed_output[:self.max_output_length] + "\n... (output truncated)"
                
            figures = self._capture_figures()
            dataframes = self._find_dataframes(locals_dict)
            
            output = None
            if 'result' in locals_dict:
                output = locals_dict['result']
            elif len(dataframes) > 0:
                output = list(dataframes.values())[0]  # Return first DataFrame if present
            
            return ExecutionResult(
                status=True,
                output=output,
                generated_figures=figures,
                printed_output=printed_output,
                dataframes=dataframes
            )
            
        except Exception as e:
            if self.capture_prints:
                self._restore_output()
            return ExecutionResult(
                status=False,
                output=None,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )

    def execute_multiple(self, codes: List[str], data: Any = None) -> List[ExecutionResult]:
        """Execute multiple code blocks sequentially"""
        results = []
        current_data = data
        
        for code in codes:
            result = self.execute(code, current_data)
            results.append(result)
            
            # Update data for next iteration if current execution produced a DataFrame
            if result.status and result.dataframes:
                current_data = list(result.dataframes.values())[0]
                
        return results