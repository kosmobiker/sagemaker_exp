import requests

#provide values
restapi_id = ''
region = 'eu-west-1'
stage_name = ''

INVOKE_URL = f"https://{restapi_id}.execute-api.{region}.amazonaws.com/{stage_name}"


test_toxic_comment = """I’m a bit sleepy tonight but when I wake up I’m going death con 3 On JEWISH PEOPLE.
The funny thing is I actually can’t be Anti Semitic because black people are actually Jew 
also You guys have toyed with me and tried to black ball anyone whoever opposes your agenda."""

input_data = {"inputs": test_toxic_comment}


def main():
    return requests.post(INVOKE_URL, json=input_data)


if __name__ == "__main__":
    for _ in range(1000):
        main()
