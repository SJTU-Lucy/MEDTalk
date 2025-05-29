import os
import base64
from openai import OpenAI

api_key = "sk-lLywoHlHcl7ecr5QItiRby6Ry2IdV9PSrYG56KSpnO5ILTrO"
base_url = "https://xiaoai.plus/v1"
model_name = "gemini-2.0-flash-thinking-exp"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_instruction(image_path=None, text=None):
    print(f"model: {model_name} | image_path: {image_path} | text: {text}")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    system_text = "Please briefly describe the facial expressions of the people in the picture, " \
                  "including the state of their mouths, eyes, and eyebrows, and speculate on their possible emotions."
    messages = []

    messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": []})
    if image_path is not None:
        base64_image = encode_image(image_path)
        messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
    if text is not None:
        messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": text
                    })
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return completion


if  __name__ == "__main__":
    image_path = "RAVDESS/image"
    text_path = "RAVDESS/text"
    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)
        text_file = os.path.join(text_path, file[:-4] + ".txt")
        print(image_file, text_file)
        try:
            completion = get_instruction(
                image_path=image_file,
                text="please answer in brief yet vivid words within two sentences.")
            content = completion.choices[0].message.content
            print(f"assisant: {content}")
        except Exception as e:
            print(f"发生错误: {e}")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(content)
            f.close()
