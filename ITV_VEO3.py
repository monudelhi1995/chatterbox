import time
from google import genai

client = genai.Client()

prompt = "Panning wide shot of a calico kitten sleeping in the sunshine"

# Step 1: Generate an image with Imagen.
imagen = client.models.generate_images(
    model="imagen-3.0-generate-002",
    prompt=prompt,
)

# Step 2: Generate video with Veo 3 using the image.
operation = client.models.generate_videos(
    model="veo-3.0-generate-preview",
    prompt=prompt,
    image=imagen.generated_images[0].image,
)

# Poll the operation status until the video is ready.
while not operation.done:
    print("Waiting for video generation to complete...")
    time.sleep(10)
    operation = client.operations.get(operation)

# Download the video.
video = operation.response.generated_videos[0]
client.files.download(file=video.video)
video.video.save("veo3_with_image_input.mp4")
print("Generated video saved to veo3_with_image_input.mp4")