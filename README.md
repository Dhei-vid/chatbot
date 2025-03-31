# Building a chatbot using Python and Huggingface opensource transformers libraries - https://huggingface.co/models

## Installing transformers library
 ```
python3 -m pip install transformers==4.30.2 torch
or
python3.11 -m pip install transformers torch
```

## Integrating a Chatbot into a web interface
 ```
  python3.11 -m pip install flask
  python3.11 -m pip install flask_cors
```

### After writing the code, in new terminal run
```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}' 127.0.0.1:5000/chatbot
```
