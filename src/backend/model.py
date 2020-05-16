from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
	
class Host:
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
		self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
		self.chat_history_ids = None
		
	def reply(self, prompt, seed=False):
		prompt_tokens = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
		bot_input_ids = torch.cat([self.chat_history_ids, prompt_tokens], dim=-1) if seed else prompt_tokens
		
		self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
		model_output = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
		
		return model_output
		
class Guest:
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
		self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
		self.chat_history_ids = None
		
	def reply(self, prompt, seed=False):
		prompt_tokens = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
		bot_input_ids = torch.cat([self.chat_history_ids, prompt_tokens], dim=-1) if seed else prompt_tokens
		
		self.chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
		model_output = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
		
		return model_output		
		
host = Host()
guest = Guest()

reply = "Hello, welcome to the podcast! Can you introduce yourself?"
print (reply)

for _ in range(10):
	reply = host.reply(reply)
	print (reply)
	reply = guest.reply(reply)
	print (reply)