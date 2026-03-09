from ai.assistant import run_assistant

print("AI Assistant Started")

while True:

    user_input = input("You: ")

    if user_input == "exit":
        break

    reply = run_assistant(user_input)

    print("AI:", reply)