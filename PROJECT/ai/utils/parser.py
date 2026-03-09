def parse_details(text):

    data = {}
    current_key = None

    for line in text.split("\n"):

        line = line.rstrip()

        if ":" in line:
            key, value = line.split(":", 1)

            key = key.strip().lower()
            value = value.strip()

            current_key = key
            data[current_key] = value

        elif current_key:
            data[current_key] += "\n" + line.strip()

    return data