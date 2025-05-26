import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Load environment variables from .env file
load_dotenv()

class SAPLogsInteractiveQuery:
    def __init__(self, excel_path, api_key=None):
        """
        Initialize the interactive SAP Logs Query System
        
        Args:
            excel_path (str): Path to the Excel file containing SAP logs
            api_key (str, optional): OpenAI API key. If None, will try to get from environment variable
        """
        # Load the SAP logs data
        self.df = pd.read_excel(excel_path)
        
        # Clean column names (remove leading/trailing spaces)
        self.df.columns = [col.strip() for col in self.df.columns]
        
        # Print dataset information
        self._print_dataset_info()
        
        # Get API key from environment variable if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Initialize the LLM with OpenAI
        self.llm = OpenAI(api_token=api_key)
        
        # Initialize PandasAI with the LLM
        self.pandas_ai = SmartDataframe(self.df, config={"llm": self.llm})
    
    def _print_dataset_info(self):
        """Print basic information about the dataset"""
        print(f"✅ Loaded SAP logs with {len(self.df)} records")
        print(f"✅ Columns: {len(self.df.columns)} columns identified\n")
        
        # Data types overview
        num_numeric = len(self.df.select_dtypes(include=['number']).columns)
        num_datetime = len(self.df.select_dtypes(include=['datetime']).columns)
        num_object = len(self.df.select_dtypes(include=['object']).columns)
        
        print(f"Data types: {num_numeric} numeric, {num_datetime} datetime, {num_object} text/categorical")
        
        # Missing values check
        missing = self.df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if len(cols_with_missing) > 0:
            pass
            # print("\n⚠️ Columns with missing values:")
            # for col, count in cols_with_missing.items():
            #     print(f"  - {col}: {count} missing values ({count/len(self.df):.1%})")
        else:
            print("\n✅ No missing values detected in the dataset")
    
    def query(self, question):
        """
        Query the SAP logs using natural language
        
        Args:
            question (str): Natural language question about the SAP logs
            
        Returns:
            The response from SmartDataframe
        """
        print(f"\n📝 Query: {question}")
        try:
            response = self.pandas_ai.chat(question)
            return response
        except Exception as e:
            print(f"Error during query execution: {str(e)}")
            return None
    
    def explore_column(self, column_name):
        """
        Explore a specific column with basic statistics and visualization
        
        Args:
            column_name (str): Name of the column to explore
        """
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found. Available columns are:")
            print("\n".join(f"- {col}" for col in self.df.columns))
            return
        
        print(f"\n📊 Exploring column: {column_name}")
        
        # Basic statistics
        print("\nBasic statistics:")
        if pd.api.types.is_numeric_dtype(self.df[column_name]):
            stats = self.df[column_name].describe()
            for stat_name, value in stats.items():
                print(f"- {stat_name}: {value:.2f}")
            
            # Visualization for numeric columns
            plt.figure(figsize=(10, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(self.df[column_name].dropna(), kde=True)
            plt.title(f'Distribution of {column_name}')
            plt.tight_layout()
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(y=self.df[column_name].dropna())
            plt.title(f'Boxplot of {column_name}')
            plt.tight_layout()
            
            plt.show()
        else:
            # For categorical columns
            value_counts = self.df[column_name].value_counts()
            print(f"- Unique values: {len(value_counts)}")
            print("\nTop values:")
            for value, count in value_counts.head(10).items():
                print(f"- {value}: {count} ({count/len(self.df):.1%})")
            
            # Visualization for categorical columns
            if len(value_counts) <= 20:  # Only plot if not too many categories
                plt.figure(figsize=(10, 6))
                sns.countplot(y=self.df[column_name], order=value_counts.index[:20])
                plt.title(f'Distribution of {column_name}')
                plt.tight_layout()
                plt.show()
    
    def performance_dashboard(self):
        """
        Create a dashboard of key SAP performance metrics
        """
        if 'Response Time in ms' not in self.df.columns or 'Server' not in self.df.columns:
            print("Required columns not found for performance dashboard")
            return
        
        print("\n📈 SAP Performance Dashboard")
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # 1. Response time by server
        plt.subplot(2, 2, 1)
        server_perf = self.df.groupby('Server')['Response Time in ms'].mean().sort_values(ascending=False)
        sns.barplot(x=server_perf.values, y=server_perf.index)
        plt.title('Average Response Time by Server')
        plt.xlabel('Response Time (ms)')
        
        # 2. DB performance metrics
        plt.subplot(2, 2, 2)
        db_metrics = ['DB request time ms', 'Direct read request time ms', 
                   'Sequential read request time ms', 'Update request time ms']
        db_metrics = [m for m in db_metrics if m in self.df.columns]
        
        if db_metrics:
            db_perf = self.df[db_metrics].mean()
            sns.barplot(x=db_perf.values, y=db_perf.index)
            plt.title('Average Database Performance Metrics')
            plt.xlabel('Time (ms)')
        
        # 3. Memory usage distribution
        plt.subplot(2, 2, 3)
        memory_cols = ['Maximum memory roll KB', 'Total allocated page memory KB', 
                     'Maximum extended memory in task KB']
        memory_cols = [m for m in memory_cols if m in self.df.columns]
        
        if memory_cols and len(memory_cols) > 0:
            for col in memory_cols:
                sns.kdeplot(self.df[col].dropna(), label=col)
            plt.title('Memory Usage Distribution')
            plt.xlabel('Memory (KB)')
            plt.legend()
        
        # 4. CPU vs Response Time scatter
        plt.subplot(2, 2, 4)
        if 'CPU time ms' in self.df.columns and 'Response Time in ms' in self.df.columns:
            sns.scatterplot(x='CPU time ms', y='Response Time in ms', data=self.df, alpha=0.5)
            plt.title('CPU Time vs Response Time')
            plt.xlabel('CPU Time (ms)')
            plt.ylabel('Response Time (ms)')
        
        plt.tight_layout()
        plt.show()
    
    def run_interactive_session(self):
        """
        Run an interactive query session where the user can input questions
        """
        print("\n🤖 SAP Logs Interactive Query Session")
        print("Type 'exit' to quit, 'help' for commands, or enter your question")
        
        while True:
            user_input = input("\n> ")
            
            if user_input.lower() == 'exit':
                print("Exiting interactive session")
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit: Exit the interactive session")
                print("- help: Show this help message")
                print("- columns: Show all column names")
                print("- head: Show the first 5 rows of data")
                print("- explore [column]: Show statistics for a specific column")
                print("- dashboard: Show a performance dashboard")
                print("Or type any natural language question to query the data")
            elif user_input.lower() == 'columns':
                for col in self.df.columns:
                    print(f"- {col}")
            elif user_input.lower() == 'head':
                display(self.df.head())
            elif user_input.lower().startswith('explore '):
                column = user_input[8:].strip()
                self.explore_column(column)
            elif user_input.lower() == 'dashboard':
                self.performance_dashboard()
            else:
                response = self.query(user_input)
                print("\nResult:")
                print(response)


# Example usage
if __name__ == "__main__":
    excel_path = "sap_logs.xlsx"
    sap_query = SAPLogsInteractiveQuery(excel_path)
    
    print("\n--- Example Queries ---")
    
    response = sap_query.query("which program have highest Roll outs value")
    print("\nResult:")
    print(response)