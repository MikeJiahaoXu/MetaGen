"""
Generate synthetic data for the metasurface design process, which can be used to do the supervised fine-tuning of the LLM.
"""
import os  
from openai import AzureOpenAI
import json
import re
from tqdm import tqdm
import argparse


ENDPOINT = os.getenv("ENDPOINT_URL", "https://generate-dataset.openai.azure.com/")
DEPLOY = os.getenv("DEPLOYMENT_NAME", "gpt-4")  
APIKEY = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")  

MATERIAL = ["Rogers RO3006 (loss free)", "Rogers RO3010 (loss free)", "Rogers RO3203 (loss free)", "Rogers RO3210 (loss free)", "Rogers RO4003C (loss free)", "Rogers RT-duroid 5870 (loss free)", "Rogers RT-duroid 5880 (loss free)", "Rogers RT-duroid 6002 (loss free)", "Rogers TMM 10 (loss free)", "Rogers TMM 10i (loss free)", "Rogers TMM 3 (loss free)", "Rogers TMM 4 (loss free)", "Rogers TMM 6 (loss free)", "FR-4 (loss free)", "Rogers RO3003 (loss free)"]
# Define the template for the synthetic data, few-shot strategy
TEMPLATE = [
{
    "role": "system",
    "content": "You are an assistant synthesizing training data.\nTask description:\nGenerate a multi-turn dialogue training dataset of reflective metasurface design process. \nThe data you generate should follow the format where the user provides initial design intent, and the assistant asks questions to gather specific details about the metasurface. \nThe assistant content should try to extract the following information through the dialogue: working frequency band, corresponding frequency band width, substrate material, unit size, and height. After all the above information is obtained, the assistant need to provide the corresponding EM responses description vector and conditon vector of the metasurface based on the conversation.\nThe user content should answer the question in the above assistant content, and the specific answer is provided in the input json.\n\nTips:\n1. Start the dialogue with the user expressing their intent to design a reflective metasurface.\n2. Ask the user about more detailed information, you are free to decide the order and manner of your questions. The answers to the question are values provided in the input JSON data. If the user provides all the required information directly, no further questions are required.\n3.(important) After all the information is obtained, the last round of dialogue is completed by assistant. This round of dialogue provides key-value pairs in the input data directly and is wrapped with <PARAM/></PARAM>.\n4. The generated multi-round dialogue can be more varied in form, for example, it is not necessary to ask the required parameters in a specific order, and users can take the initiative to provide some parameters."
},
{
    "role": "user",
    "content": "{\"peak_freq\": [\"73.1GHz\"], \"band_width\": [\"1.7GHz\"], \"material\": \"Rogers TMM 4 (loss free)\", \"size\": \"3840.0um\", \"height\": \"1000.0um\", \"Imag(S11)\": [-0.096, -0.145, -0.177, -0.196, -0.208, -0.213, -0.213, -0.21, -0.203, -0.194, -0.183, -0.17, -0.156, -0.139, -0.122, -0.103, -0.083, -0.062, -0.04, -0.018, 0.005, 0.029, 0.053, 0.078, 0.102, 0.127, 0.152, 0.177, 0.202, 0.226, 0.25, 0.274, 0.296, 0.318, 0.339, 0.358, 0.375, 0.389, 0.397, 0.395, 0.375, 0.321, 0.197, -0.055, -0.436, -0.681, -0.577, -0.346, -0.152, -0.016, 0.079, 0.145, 0.192, 0.226, 0.242, 0.133, -0.686, 0.037, 0.154, 0.189, 0.202, 0.205, 0.202, 0.195, 0.184, 0.17, 0.155], \"Real(S11)\": [0.61, 0.541, 0.479, 0.422, 0.372, 0.326, 0.285, 0.248, 0.214, 0.183, 0.154, 0.129, 0.105, 0.084, 0.065, 0.048, 0.033, 0.021, 0.01, 0.002, -0.004, -0.008, -0.01, -0.009, -0.006, -0.001, 0.006, 0.017, 0.03, 0.045, 0.064, 0.086, 0.111, 0.141, 0.175, 0.215, 0.262, 0.318, 0.385, 0.467, 0.569, 0.696, 0.842, 0.96, 0.887, 0.506, 0.122, -0.045, -0.071, -0.039, 0.014, 0.077, 0.148, 0.236, 0.37, 0.678, 0.003, -0.056, 0.082, 0.165, 0.225, 0.274, 0.315, 0.352, 0.385, 0.416, 0.445], \"freq\": [60.0, 60.3, 60.6, 60.9, 61.2, 61.5, 61.8, 62.1, 62.4, 62.7, 63.0, 63.3, 63.6, 63.9, 64.2, 64.5, 64.8, 65.1, 65.4, 65.7, 66.0, 66.3, 66.6, 66.9, 67.2, 67.5, 67.8, 68.1, 68.4, 68.7, 69.0, 69.3, 69.6, 69.9, 70.2, 70.5, 70.8, 71.1, 71.4, 71.7, 72.0, 72.3, 72.6, 72.9, 73.2, 73.5, 73.8, 74.1, 74.4, 74.7, 75.0, 75.3, 75.6, 75.9, 76.2, 76.5, 76.8, 77.1, 77.4, 77.7, 78.0, 78.3, 78.6, 78.9, 79.2, 79.5, 79.8]}"
},
{
    "role": "assistant",
    "content": "[\n  {\"role\": \"user\", \"content\": \"I want to design a reflective metasurface.\"},\n  {\"role\": \"assistant\", \"content\": \"Ok, what frequency band do you want it to work on?\"},\n  {\"role\": \"user\", \"content\": \"I want it to work on 73.1GHz with a bandwidth of 1.7GHz.\"},\n  {\"role\": \"assistant\", \"content\": \"What material should the metasurface substrate be made of?\"},\n  {\"role\": \"user\", \"content\": \"The material should be Rogers TMM 4 (loss free), and I want it height is 1000.0um.\"},\n  {\"role\": \"assistant\", \"content\": \"What should be the size of the metasurface unit?\"},\n  {\"role\": \"user\", \"content\": \"The size should be 3840.0um.\"},\n  {\"role\": \"assistant\", \"content\": \" <PARAM/>{\"material\": \"Rogers TMM 4 (loss free)\", \"size\": \"3840.0um\", \"height\": \"1000.0um\", \"Imag(S11)\": [-0.096, -0.145, -0.177, -0.196, -0.208, -0.213, -0.213, -0.21, -0.203, -0.194, -0.183, -0.17, -0.156, -0.139, -0.122, -0.103, -0.083, -0.062, -0.04, -0.018, 0.005, 0.029, 0.053, 0.078, 0.102, 0.127, 0.152, 0.177, 0.202, 0.226, 0.25, 0.274, 0.296, 0.318, 0.339, 0.358, 0.375, 0.389, 0.397, 0.395, 0.375, 0.321, 0.197, -0.055, -0.436, -0.681, -0.577, -0.346, -0.152, -0.016, 0.079, 0.145, 0.192, 0.226, 0.242, 0.133, -0.686, 0.037, 0.154, 0.189, 0.202, 0.205, 0.202, 0.195, 0.184, 0.17, 0.155], \"Real(S11)\": [0.61, 0.541, 0.479, 0.422, 0.372, 0.326, 0.285, 0.248, 0.214, 0.183, 0.154, 0.129, 0.105, 0.084, 0.065, 0.048, 0.033, 0.021, 0.01, 0.002, -0.004, -0.008, -0.01, -0.009, -0.006, -0.001, 0.006, 0.017, 0.03, 0.045, 0.064, 0.086, 0.111, 0.141, 0.175, 0.215, 0.262, 0.318, 0.385, 0.467, 0.569, 0.696, 0.842, 0.96, 0.887, 0.506, 0.122, -0.045, -0.071, -0.039, 0.014, 0.077, 0.148, 0.236, 0.37, 0.678, 0.003, -0.056, 0.082, 0.165, 0.225, 0.274, 0.315, 0.352, 0.385, 0.416, 0.445], \"freq\": [60.0, 60.3, 60.6, 60.9, 61.2, 61.5, 61.8, 62.1, 62.4, 62.7, 63.0, 63.3, 63.6, 63.9, 64.2, 64.5, 64.8, 65.1, 65.4, 65.7, 66.0, 66.3, 66.6, 66.9, 67.2, 67.5, 67.8, 68.1, 68.4, 68.7, 69.0, 69.3, 69.6, 69.9, 70.2, 70.5, 70.8, 71.1, 71.4, 71.7, 72.0, 72.3, 72.6, 72.9, 73.2, 73.5, 73.8, 74.1, 74.4, 74.7, 75.0, 75.3, 75.6, 75.9, 76.2, 76.5, 76.8, 77.1, 77.4, 77.7, 78.0, 78.3, 78.6, 78.9, 79.2, 79.5, 79.8]} </PARAM>\n\"}\n]"
},
{
    "role": "user",
    "content": "{\"peak_freq\": [\"4.6GHz\", \"6.6GHz\", \"8.92GHz\"], \"band_width\": [\"0.24GHz\", \"0.2GHz\", \"0.16GHz\"], \"material\": \"Rogers TMM 3 (loss free)\", \"size\": \"109440.0um\", \"height\": \"1200.0um\", \"Imag(S11)\": [-0.019, -0.021, -0.016, -0.001, 0.065, 0.002, 0.018, 0.096, 0.439, 0.034, 0.047, 0.072, 0.099, 0.131, 0.177, 0.197, 0.18, 0.252, 0.323, 0.41, 0.535, 0.679, 0.212, -0.255, -0.132, -0.018, 0.066, 0.142, 0.236, 0.159, 0.248, 0.074, 0.208, 0.139, 0.247, 0.27, 0.267, 0.247, -0.101, -0.554, -0.094, -0.143, 0.079, 0.106, 0.102, -0.153, 0.011, -0.046, -0.024, -0.178, 0.021, 0.017, -0.403, -0.14, -0.336, 0.064, -0.106, -0.433, -0.3, 0.077, -0.024, -0.06, -0.002, -0.092, -0.119, -0.224, -0.109], \"Real(S11)\": [-0.092, -0.119, -0.15, -0.194, -0.31, 0.033, -0.155, -0.26, 0.17, -0.013, -0.091, -0.124, -0.143, -0.154, -0.154, -0.157, -0.091, -0.156, -0.14, -0.102, -0.009, 0.283, 0.824, 0.328, 0.064, -0.01, -0.027, -0.017, 0.173, 0.022, 0.087, 0.002, 0.069, 0.051, 0.155, 0.234, 0.315, 0.486, 0.83, 0.119, -0.096, -0.152, 0.054, 0.146, 0.256, -0.187, 0.37, 0.261, 0.212, 0.492, -0.105, 0.105, -0.098, 0.34, 0.411, -0.093, 0.258, 0.342, -0.642, -0.062, 0.095, -0.068, 0.09, 0.147, 0.124, 0.16, 0.13], \"freq\": [2.0, 2.12, 2.24, 2.36, 2.48, 2.6, 2.72, 2.84, 2.96, 3.08, 3.2, 3.32, 3.44, 3.56, 3.68, 3.8, 3.92, 4.04, 4.16, 4.28, 4.4, 4.52, 4.64, 4.76, 4.88, 5.0, 5.12, 5.24, 5.36, 5.48, 5.6, 5.72, 5.84, 5.96, 6.08, 6.2, 6.32, 6.44, 6.56, 6.68, 6.8, 6.92, 7.04, 7.16, 7.28, 7.4, 7.52, 7.64, 7.76, 7.88, 8.0, 8.12, 8.24, 8.36, 8.48, 8.6, 8.72, 8.84, 8.96, 9.08, 9.2, 9.32, 9.44, 9.56, 9.68, 9.8, 9.92]}"
},
{
    "role": "assistant",
    "content": "[\n  {\"role\": \"user\", \"content\": \"I want to design a reflective metasurface which worked on 4.6GHz and 8.92GHz and I want the unit size is 109440.0um.\"},\n  {\"role\": \"assistant\", \"content\": \"Sure, what band width and substrate do you want?\"},\n  {\"role\": \"user\", \"content\": \"The band width is 0.24GHz and 0.16GHz. I want to use Rogers TMM 3 (loss free), and I want the height of it is 1200.0um.\"},\n  {\"role\": \"assistant\", \"content\": \" <PARAM/>{\"material\": \"Rogers TMM 3 (loss free)\", \"size\": \"109440.0um\", \"height\": \"1200.0um\", \"Imag(S11)\": [-0.019, -0.021, -0.016, -0.001, 0.065, 0.002, 0.018, 0.096, 0.439, 0.034, 0.047, 0.072, 0.099, 0.131, 0.177, 0.197, 0.18, 0.252, 0.323, 0.41, 0.535, 0.679, 0.212, -0.255, -0.132, -0.018, 0.066, 0.142, 0.236, 0.159, 0.248, 0.074, 0.208, 0.139, 0.247, 0.27, 0.267, 0.247, -0.101, -0.554, -0.094, -0.143, 0.079, 0.106, 0.102, -0.153, 0.011, -0.046, -0.024, -0.178, 0.021, 0.017, -0.403, -0.14, -0.336, 0.064, -0.106, -0.433, -0.3, 0.077, -0.024, -0.06, -0.002, -0.092, -0.119, -0.224, -0.109], \"Real(S11)\": [-0.092, -0.119, -0.15, -0.194, -0.31, 0.033, -0.155, -0.26, 0.17, -0.013, -0.091, -0.124, -0.143, -0.154, -0.154, -0.157, -0.091, -0.156, -0.14, -0.102, -0.009, 0.283, 0.824, 0.328, 0.064, -0.01, -0.027, -0.017, 0.173, 0.022, 0.087, 0.002, 0.069, 0.051, 0.155, 0.234, 0.315, 0.486, 0.83, 0.119, -0.096, -0.152, 0.054, 0.146, 0.256, -0.187, 0.37, 0.261, 0.212, 0.492, -0.105, 0.105, -0.098, 0.34, 0.411, -0.093, 0.258, 0.342, -0.642, -0.062, 0.095, -0.068, 0.09, 0.147, 0.124, 0.16, 0.13], \"freq\": [2.0, 2.12, 2.24, 2.36, 2.48, 2.6, 2.72, 2.84, 2.96, 3.08, 3.2, 3.32, 3.44, 3.56, 3.68, 3.8, 3.92, 4.04, 4.16, 4.28, 4.4, 4.52, 4.64, 4.76, 4.88, 5.0, 5.12, 5.24, 5.36, 5.48, 5.6, 5.72, 5.84, 5.96, 6.08, 6.2, 6.32, 6.44, 6.56, 6.68, 6.8, 6.92, 7.04, 7.16, 7.28, 7.4, 7.52, 7.64, 7.76, 7.88, 8.0, 8.12, 8.24, 8.36, 8.48, 8.6, 8.72, 8.84, 8.96, 9.08, 9.2, 9.32, 9.44, 9.56, 9.68, 9.8, 9.92]} </PARAM>\"}\n]"
},
]  



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', help='path to the metasurface dataset, including s-params and other geometric params', type=str, default="metadata.jsonl")
	parser.add_argument('--output_path', help='path to the metasurface dataset, including s-params and other geometric params', type=str, default="output_final.jsonl")
	args = parser.parse_args()

	# Create an instance of the AzureOpenAI class
	client = AzureOpenAI(  
		azure_endpoint=ENDPOINT,  
		api_key=APIKEY,  
		api_version="2024-05-01-preview",  
	)
	with open(args.input_path, 'r') as file:
		for line in tqdm(file):
			data = json.loads(line)
			new_format = {
				'peak_freq': [f"{item}GHz" for item in data['work_freq']], 
				'band_width': [f"{round(item, 3)}GHz" for item in data['band_width']],
				'material': MATERIAL[data['condition'][2]], 
				'size': f'{str(64 * data["condition"][0])}um', 
				'height': f'{data["condition"][1]}um', 
				'Imag(S11)': data['s_param'][603:804][::3], 
				'Real(S11)': data['s_param'][402:603][::3], 
				'freq': data['condition'][3:][::3]
			}
			json_str = json.dumps(new_format)
			chat_prompt = TEMPLATE + [{
				"role": "user",
				"content": json_str
			}]
			speech_result = chat_prompt
			completion = client.chat.completions.create(  
				model=DEPLOY,  
				messages=speech_result,  
				# past_messages=10,  
				max_tokens=1600,  
				temperature=0.76,  
				top_p=0.9,  
				frequency_penalty=0,  
				presence_penalty=0,  
				stop=None,  
				stream=False  
			)  
			a = completion.to_json()
			content = json.loads(a)['choices'][0]['message']['content']
			pattern = r'(<PARAM/>)(.*?)(</PARAM>)'
			matches = re.finditer(pattern, content)

			for match in matches:
				param_content = match.group(2)
				escaped_content = param_content.replace("\"", "\\\"")
				content = content.replace(param_content, escaped_content)

			output_str = f'{{"messages": {content}}}\n'

			with open(args.output_path, "a") as file:
				file.write(output_str)
			

