import os
from dotenv import load_dotenv, find_dotenv
import chromadb
import openai
from openai import OpenAI
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadbx import UUIDGenerator
import json
import sqlite3
from streamlit import logger
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
app_loger = logger.get_logger("MemoryAIAPLog")
app_loger.log(f"sqlite version: {sqlite3.sqlite_version}")

# # Ensure python uses its own sqlite3 instead of a system one
# print("SQLite3 version used by Python:", sqlite3.sqlite_version)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
OpenAI_API_KEY = os.getenv("OpenAI_APIKEY")


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_content: Documents) -> Embeddings:
        client = OpenAI(api_key= OpenAI_API_KEY)
        embeddings = embedding_func = client.embeddings.create(
            model='text-embedding-3-large',
            input= input_content
            ).data[0].embedding
        return embeddings

# Set up the client
client = OpenAI(api_key= OpenAI_API_KEY)

# Set up database
def set_up_database():
    """Set up the database."""
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="test", metadata={"hnsw:space": "cosine"}, embedding_function=MyEmbeddingFunction())
    return collection

def add_document(collection, text, metadata):
    """Add a document to the database."""
    collection.add(
    documents=[text],
    ids=UUIDGenerator(1),
    metadatas=metadata
    )

def retrieve_document(collection, document, metadata):
    query_where = {"$and": [{"fruit": "pineapple"}, {"climate": "tropical"}]}
    query_where_document = {"$contains": document}
    select_ids = collection.get(where_document=query_where_document, where=query_where, include=[])
    # batch = collection.get(include=["metadatas", "documents", "embeddings"], limit=10, offset=0, where=query_where,
    #                 where_document=query_where_document)

    # batch_size = 10
    # for i in range(0, len(select_ids["ids"]), batch_size):
    #     batch = collection.get(include=["metadatas", "documents", "embeddings"], limit=batch_size, offset=i, where=query_where,
    #                     where_document=query_where_document)
    #     newCol.add(ids=batch["ids"], documents=batch["documents"], metadatas=batch["metadatas"],
    #             embeddings=batch["embeddings"])
    # print(newCol.get(offset=0, limit=10))

    result = collection.query(
    query_texts=document,
    n_results=10,
    where=metadata,
    #where_document={"$contains":document}
    )
    print(result)
    return result['documents']

def augmentation_process(documents):
    """Aggregate and augment the documents."""
   
    prompt = f"""Given the following list of documents, {documents}, 
    create a single cohesive paragraph that summarizes the key information. 
    Ensure the paragraph is clear and accessible for visually impaired individuals and those with early Alzheimer's. 
    Focus on simplicity and clarity, connecting the main ideas smoothly without excessive detail."""
   
    result = client.chat.completions.create(
            model="gpt-4o", # model to send to the proxy
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    print(result.choices[0].message.content)

def set_up_content(collection):
    """put seed data into the database"""

    add_document(
        collection, 
        "Barton Hall is an on-campus field house on the campus of Cornell University in Ithaca, New York. It is the site of the school's indoor track facilities, ROTC offices and classes, and Cornell Police. For a long time, Barton Hall was the largest unpillared room in existence.", 
        {"location": "Barton Hall", "type": "campus_building", "image_url":"https://venuellama.com/wp-content/uploads/formidable/18/bh-1-4.jpg"}
    )

    add_document(
        collection, 
        "This building holds many stressful memories of exams. My biology exam was in here, for some reason the room was freezing and all the questions were really long and tricky.",
        {"location": "Barton Hall", "type": "personal_memory", "class_year": "junior", "semester": "Fall 2023", "memory": "sad"}
    )

    add_document(
        collection, 
        "The giant space was perfect for club fest! I learned about many cool clubs here and made many friends. The club performances were very inspiring. My first impression of the student body of Cornell couldn't be better.",
        {"location": "Barton Hall", "type": "personal_memory", "class_year": "freshman", "memory": "happy"}
    );


    add_document(
        collection,
        "Libe Slope is a steep hillside on Cornell's campus, offering breathtaking views of Cayuga Lake. It is a popular spot for students to relax and enjoy the scenery.",
        {"location": "Libe Slope", "type": "campus_landmark", "image_url": "https://i1.wp.com/cornellsun.com/wp-content/uploads/2021/10/HR-1532-scaled.jpg?resize=1536%2C1024&ssl=1"}
    )

    add_document(
        collection, 
        "My first time sledding was on this slope. I had so much fun and the slope was just perfectly covered in a thick blanket of snow. My hands froze to the bone from getting wet from the snow but it was still fun. My favorite was sledding racing with my friends. I won of course!",
        {"location": "Libe Slope", "season": "winter", "type": "personal_memory", "class_year": "sophomore", "memory": "happy", "setting": "outdoors"}
    )

    add_document(
        collection, 
        "The sunset from libe slope was very beautiful and scenic. I took many pictures because the sky was purple and pink which I have never seen before. The sunsets from the slope are my favorite. I saw some hot air balloons pass in the far distance too.",
        {"season": "spring", "type": "personal_memory", "memory": "happy", "location": "Libe Slope", "setting": "nature"}
    )

    add_document(
        collection,
        "Olin Library is a central academic hub at Cornell, housing thousands of books and providing study spaces for students. It is known for its extensive research resources.",
        {"location": "Olin Library", "type": "campus_building", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Cornell_Olin_Library_2.jpg/760px-Mapcarta.jpg"}
    )

    add_document(
        collection, 
        "Olin Library is a nice but busy library to study in between classes. I got the code with access to the grad study lounge because it is less busy. I like studying here. I always like to get a drink from the cafe in the library.",
        {"location": "Olin Library", "type": "personal_memory", "memory": "busy", "setting": "study"}
    )

    add_document(
        collection,
        "Goldwin Smith Hall is home to Cornell's College of Arts and Sciences. It features classrooms, lecture halls, and offices for humanities departments.",
        {"location": "Goldwin Smith Hall", "type": "campus_building", "image_url": "https://i2.wp.com/cornellsun.com/wp-content/uploads/2018/10/Pg.-1-Goldwin-Smith-Hall.jpg?resize=1170%2C781&ssl=1x"}
    )

    add_document(
        collection, 
        "I had a class in Goldwin Smith Hall for my English writing seminar with Professor Cynthia Robinson. It was a fun class that I learned a lot from. I did not like writing though. Goldwin Smith was always a good place for pausing between classes or club meetings.",
        {"location": "Goldwin Smith Hall", "class_year": "freshman", "type": "personal_memory", "setting": "class", "memory": "good"}
    )

    add_document(
        collection, 
        "Cynthia Robinson was a professor at Cornell. I took her English writing seminar course. She was stern but had an interesting sense of humor. She frequently called on us which was not fun. She also gave confusing feedback on essays. But she had a cute pet rabbit.",
        {"type": "person", "name": "Cynthia Robinson", "relationship": "professor", "image_url": "https://people.as.cornell.edu/sites/default/files/styles/person_image/public/cynthia-robinson.jpg"}
    )

    add_document(
        collection,
        "Willard Straight Hall is a student union building at Cornell, offering dining options, study spaces, and venues for student activities and events.",
        {"location": "Willard Straight Hall", "type": "campus_building", "image_url": "https://en.wikipedia.org/wiki/File:Willard_Straight_Hall,_Cornell_University.jpg"}
    )

    add_document(
        collection, 
        "The dining hall Okenshields was in here. One evening they served sesame balls and it was delicious. I went back but did not see them again. I think the food is ok otherwise. They also serve clam chowder sometimes that I love.",
        {"location": "Willard Straight Hall", "setting": "dining hall", "type": "personal_memory", "memory": "happy", "activity": "food", "class_year": "junior"}
    )

    add_document(
        collection,
        "Baker Tower is a residential building on Cornell's West Campus, offering housing for upper-level and graduate students.",
        {"location": "Baker Tower", "type": "campus_building", "image_url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Baker_Tower%2C_Cornell_University.jpg"}
    )

    add_document(
        collection, 
        "Baker Tower was my dorm. My room was at the very top floor which was so exhausting to walk up to. But my room was a single with no roommates so I did not have issues and organized my room nicely to my liking. The floormates were all nice people but someone always left a mess in the bathroom.",
        {"location": "Baker Tower", "class_year": "sophomore", "setting": "dorm", "type": "personal_memory", "memory": "not good"}
    )

    add_document(
        collection,
        "Beebe Lake is a serene body of water near Cornell's North Campus, surrounded by walking trails and offering a peaceful retreat for students and visitors.",
        {"location": "Beebe Lake", "type": "natural_landmark", "image_url": "https://alumni.cornell.edu/cornellians/wp-content/uploads/sites/2/2023/07/2019_1265_039-A-1264x711.jpg"}
    )

    add_document(
        collection, 
        "I remember walking around Beebe Lake on a peaceful afternoon. The stillness of the water and the surrounding trails made it the perfect spot to clear my mind and take in nature’s beauty.",
        {"type": "personal_memory", "location": "Beebe Lake", "memory": "peaceful", "setting": "outdoors"}
    )
    add_document(
        collection,
        "Cayuga Lake is one of the Finger Lakes near Ithaca, offering opportunities for boating, fishing, and enjoying scenic views. It is a popular destination for locals and tourists.",
        {"location": "Cayuga Lake", "type": "natural_landmark", "image_url": "https://www.fllt.org/wp-content/uploads/2017/03/shoreline-aerial2_10.19.2016-005.jpg"}
    )

    add_document(
        collection, 
        "Cayuga Lake offers stunning views, and I loved spending time there, whether boating or just sitting by the shore. The sunsets over the lake are unforgettable.",
        {"type": "personal_memory", "location": "Cayuga Lake", "memory": "scenic", "setting": "outdoors"}
    )

    add_document(
        collection,
        "Stewart Park is a public park in Ithaca, featuring playgrounds, open fields, and beautiful views of Cayuga Lake. It is a great spot for picnics and relaxation.",
        {"location": "Stewart Park", "type": "nearby_attraction", "image_url": "https://images.squarespace-cdn.com/content/v1/59e8ffbc017db27f7ad313cf/1641488707364-3TLDLTDT6HTYS98L6KSE/stewart+park+trees.jpg?format=1000w"}
    )

    add_document(
        collection, 
        "Stewart Park was the perfect place to unwind on weekends. The playgrounds were always full of energy, and I loved the views of Cayuga Lake from the open fields. It became my go-to spot to relax.",
        {"type": "personal_memory", "location": "Stewart Park", "memory": "relaxing", "setting": "outdoors"}
    )

    add_document(
        collection,
        "Ithaca Target is a retail store near Cornell's campus, providing students and residents with convenient access to groceries, clothing, and household items.",
        {"location": "Ithaca Target", "type": "nearby_attraction", "image_url": "https://www.masonam.com/wp-content/uploads/2017/12/Ithaca_img1-2.jpg"}
    )

    add_document(
        collection, 
        "I remember the convenience of Ithaca Target. I always replaced my shampoo here. But one time I forgot my wallet in the car and needed to walk out of the checkout area and suspiciously walk back. It was awkward explaining the situation but the Target employees were very understanding!",
        {"type": "personal_memory", "location": "Ithaca Target", "memory": "convenient", "setting": "shopping"}
    )

    add_document(
        collection,
        "The Temple of Zeus is a popular café located in the heart of Cornell's campus. Known for its cozy atmosphere, it is a favorite spot for students and faculty to gather and enjoy coffee or snacks.",
        {"location": "Temple of Zeus", "type": "campus_landmark", "image_url": "https://as.cornell.edu/sites/default/files/styles/6_4_large/public/inline-article-images/zeus-032-6293-600px.jpg?itok=yDXY6u2E"}
    )

    add_document(
        collection, 
        "The Temple of Zeus was a popular spot to catch up with friends or grab a quick coffee between classes. But I could never study here. The tables were always full so it was not very easy to study at. The cafe only took credit card so I never frequented the place either.",
        {"type": "personal_memory", "location": "Temple of Zeus", "memory": "annoyed", "setting": "campus"}
    )

    add_document(
        collection,
        "Cornell Law School is located on the university's campus and is known for its prestigious programs and beautiful architecture. It includes lecture halls, libraries, and spaces for legal research.",
        {"location": "Cornell Law School", "type": "campus_building", "image_url": "https://www.lawschool.cornell.edu/wp-content/uploads/2020/09/UP_2016_1502_131_pano_select_1380x360_acf_cropped.jpg"}
    )

    add_document(
        collection, 
        "I spent a lot of time at Cornell Law School, especially during events and lectures. It was a beautiful building, and the space made me feel connected to the university’s academic legacy.",
        {"type": "personal_memory", "location": "Cornell Law School", "memory": "inspiring", "setting": "campus"}
    )

    add_document(
        collection,
        "Gates Hall is the home of Cornell's Computing and Information Science department. It is a cutting-edge facility featuring modern classrooms, research labs, and collaborative spaces.",
        {"location": "Gates Hall", "type": "campus_building", "image_url": "https://www.archdaily.com/565115/bill-and-melinda-gates-hall-morphosis-architects/545c3d72e58ece1e47000057-bill-and-melinda-gates-hall-morphosis-architects-image"}
    )

    add_document(
        collection, 
        "Gates Hall was a hub for all things computing and tech. There were little study pods on each floor that I used very often for my onilne meetings. It was great for privacy but very stuffy and tight feeling.",
        {"type": "personal_memory", "location": "Gates Hall", "memory": "study", "setting": "campus"}
    )

    add_document(
        collection,
        "Rhodes Hall is an academic and research building on Cornell's campus, housing programs related to engineering and computer science. It is also home to the Cornell Theory Center.",
        {"location": "Rhodes Hall", "type": "campus_building", "image_url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/Rhodes_Hall_aka_Theory_Center_at_Cornell_University.jpg"}
    )

    add_document(
        collection, 
        "Rhodes Hall was always buzzing with students from computer science and information science. I remember spending hours in the building for office hours, but almost always got the help I needed. One day I skipped all of my classes to attend every office hour session to complete an assignment before the deadline.",
        {"type": "personal_memory", "location": "Rhodes Hall", "memory": "busy", "setting": "campus"}
    )

    add_document(
        collection,
        "The Herbert F. Johnson Museum of Art is a cultural landmark on Cornell's campus, featuring an extensive collection of artwork and stunning views of Cayuga Lake.",
        {"location": "Johnson Museum of Art", "type": "campus_landmark", "image_url": "https://en.wikipedia.org/wiki/File:Johnson-museum-of-art-cornell.JPG"}
    )

    add_document(
        collection, 
        "The Johnson Museum of Art was a highlight of my time at Cornell. I loved visiting it for its art collections and the breathtaking views of Cayuga Lake. It was a great place to reflect and get inspired.",
        {"type": "personal_memory", "location": "Johnson Museum of Art", "memory": "inspiring", "setting": "cultural"}
    )

    add_document(
        collection,
        "Risley Hall is a residential building known for its creative and artistic community, housing students passionate about the arts and culture.",
        {"location": "Risley Hall", "type": "campus_building", "image_url": "https://en.wikipedia.org/wiki/File:Risley_Hall,_Cornell_University.jpg"}
    )

    add_document(
        collection, 
        "Risley Hall was known for its creative vibe. There was a lovely dining hall that I liked. They hosted a Harry Potter themed dinner once and served butter beer and other themed dishes. It tasted great.",
        {"type": "personal_memory", "location": "Risley Hall", "memory": "fun", "setting": "residential"}
    )

def set_up_seed_data(collection):
    """Set up the seed data."""
    data = json.loads(open("seed_data.json").read())["documents"]
    for d in data:
        print(d['description'])
        add_document(collection, d["description"], d["metadata"])
    

collection = set_up_database()
set_up_seed_data(collection)
# documents = retrieve_document(collection, "creative", {"location": "Risley Hall"})
# augmentation_process(documents)