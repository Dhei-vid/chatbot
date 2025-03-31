from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# using facebook/blenderbot-400M-distill because it has an open-source license and runs relatively fast.
model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# keeping track of conversation history - initialize list before conversations occur
conversation_history = []

# Fetch prompt from user
input_text ="hello, how are you doing?"

# Tokenization of user prompt and chat history - process of converting tokens to numerical representation
# In NLP tasks, you often use the encode_plus method from the tokenizer object to perform tokenization and vectorization.
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
# print(inputs)

# In doing so, you've now created a Python dictionary which contains special keywords that allow the model to properly reference its contents.


# To learn more about tokens and their associated pretrained vocabulary files, you can explore the pretrained_vocab_files_map attribute. This attribute provides a mapping of pretrained models to their corresponding vocabulary files.
tokenizer.pretrained_vocab_files_map


# generate a response use the generate() function and pass the inputs as keyword arguments
outputs = model.generate(**inputs)
print(outputs) # the result is a dictionary containing tokens

# Therefore, you just need to decode the first index of outputs to see the response in plaintext.Therefore, you just need to decode the first index of outputs to see the response in plaintext.
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)

# Alright! You've successfully had an interaction with your chatbot! You've given it a prompt, and received its response.
# update your conversation history
conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)

# Add it in a loop and run it
while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

