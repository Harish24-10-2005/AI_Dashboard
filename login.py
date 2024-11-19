

co = cohere.ClientV2(api_key='1TyaPaqTNlXozRCJYWb1RSw30nkPwPqbG8ApPLFr')

res = co.chat_stream(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "What is an LLM?"}],
)

for event in res:
    if event:
        if event.type == "content-delta":
            print(event.delta.message.content.text, end="")

