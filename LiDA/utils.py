from lida import llm
from lida import Manager, TextGenerationConfig , llm  
from llmx import  llm, TextGenerationConfig
import os 
from dotenv import load_dotenv
from cohere import Client
from IPython.display import display
import seaborn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

load_dotenv()

class lida_Model:
    def __init__(self):
        self.Cohere_API_KEY = os.getenv('cohere_api_key')
        self.cohere_client = Client(self.Cohere_API_KEY)
        self.text_gen = llm("cohere",api_key=self.Cohere_API_KEY)
        self.lida = Manager(text_gen=self.text_gen)
        self.textgen_config = TextGenerationConfig(
                    n=1,  # Number of responses
                    temperature=0.5,  # Creativity level
                    model="command-r-plus-08-2024",  # Replace with the specific Cohere model you want to use
                    use_cache=True
                )

    def summarize(self, url):
        return self.lida.summarize(url, summary_method="default", textgen_config=self.textgen_config)

    def goals(self, summary):
        return self.lida.goals(summary, n=5, textgen_config=self.textgen_config)

    def visualize(self, summary, goal,library):
        return self.lida.visualize(summary=summary, goal=goal, textgen_config=self.textgen_config, library=library)


# l = lida_Model()

# summary = l.summarize("https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv")
# goals = l.goals(summary)   
# for goal in goals:
#     display(goal)
# i = 0
# charts = l.visualize(summary=summary, goal=goals[i])  
# print(charts[0].dtype)
# if isinstance(charts, go.Figure):  # If it's a Plotly chart (Figure object)
#     print(f"**Visualization for Goal:** {goals[i].question}")
#     charts.show()

# elif isinstance(charts, list):  # If it's a list of charts (Plotly or Matplotlib)
#     print(f"**Visualizations for Goal:** {goals[i].question}")
#     for chart in charts:
#         if isinstance(chart, go.Figure):  # Plotly chart
#             chart.show()
#         elif isinstance(chart, plt.Figure):  # Matplotlib chart
#             plt.show()

# elif isinstance(charts, plt.Figure):  # If it's a Matplotlib figure
#     print(f"**Visualization for Goal:** {goals[i].question}")
#     plt.show()