GET_RELEVANT_CHUNKS_PROMPT = """
You will be given a list of transcripts from a youtube video and a customer query.
The query of the customer represents what the customer wants to know from the video.
Not every chunk of the video is going to be relevant to the user's interest.
So your task is to identify relevant chunks.
Each chunk has an 'id'.
Your response should be a list of the ids of the relevant chunks.
"""