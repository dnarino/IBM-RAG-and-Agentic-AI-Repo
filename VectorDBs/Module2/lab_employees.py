
import ssl
ssl._create_default_https_context=ssl._create_unverified_context

import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host="localhost", port=8000)


# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

collection_name = "employee_collection"
# Function to perform various types of searches within the collection
def perform_advanced_search(collection):
    try:
        # Example 1: Search for Python developers
        print("\n1. Searching for Python developers:")
        query_text = "Python developer with web development experience"
        filtered_results = collection.query(
            query_texts=query_text,
            n_results=3
        )
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
            filtered_results['ids'][0], filtered_results['documents'][0], filtered_results['distances'][0]
        )):
            metadata = filtered_results['metadatas'][0][i]
            print(f"{i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"Role: {metadata['role']}, Department: {metadata['department']}")
            print(f"Document: {document[:200]}...")

        # Filter using Metadata
        # Finding all Engineering employees:
        results = collection.get(
            where={"department": {"$eq": "Engineering"}}
        )
        print("\n3. Finding all Engineering employees:")
        print(f"Found {len(results['ids'])} Engineering employees:")
        for i in range(len(results['ids'])):
            name = results['metadatas'][i]['name']
            role = results['metadatas'][i]['role']
            exp = results['metadatas'][i]['experience']
            print(f" - {name}: {role} ({exp} years)")
        # Finding employees with 10+ years experience
        results = collection.get(
            where={"experience": {"$gte": 10}}
        )
        print("\n4. Finding employees with 10+ years experience:")
        print(f"Found {len(results['ids'])} Employees:")
        for i in range(len(results['ids'])):
            name = results['metadatas'][i]['name']
            role = results['metadatas'][i]['role']
            exp = results['metadatas'][i]['experience']
            print(f" - {name}: {role} ({exp} years experience)")
        # Finding employees in California
        results = collection.get(
            where={"location": {"$in": ["Los Angeles", "San Francisco"]}}
        )
        print("\n5. Finding employees in California:")
        print(f"Found {len(results['ids'])} employees in California:")
        for i in range(len(results['ids'])):
            name = results['metadatas'][i]['name']
            loc = results['metadatas'][i]['location']
            print(f" - {name}: {loc}")
        # === Combined Search: Similarity + Metadata Filtering ===
        # 6. Finding senior Python developers in major tech cities:
        # Query: 'senior Python developer full-stack' 
        # with filters: 8+ years experience, located in New York, San Francisco, Seattle, or Austin
        print("\n6. Finding senior Python developers in major tech cities:")
        query_text = "senior Python developer full-stack"
        results = collection.query(
            query_texts=[query_text],
            n_results=3,
            where={
                "$and": [
                    {"experience": {"$gte": 8}},
                    {"location": {"$in": ["New York", "San Francisco", "Seattle", "Austin"]}}
                ]
            }
        )
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"{i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"Role: {metadata['role']}, Department: {metadata['department']}")
            print(f"Document: {document[:200]}...")
    except Exception as error:
        print(f"Error in advanced search {error}")
def main():
    try:
        # Delete old collection if it exists to avoid dimension mismatch errors
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "A collection for storing employees",
                "hnsw:space": "cosine" # Correct way to set the metric
            },
            embedding_function=ef # Pass directly at the top level
        )
        # Defining a list of employee dictionaries
		# Each dictionary represents an individual employee with comprehensive information
        employees = [
			{
				"id": "employee_1",
				"name": "John Doe",
				"experience": 5,
				"department": "Engineering",
				"role": "Software Engineer",
				"skills": "Python, JavaScript, React, Node.js, databases",
				"location": "New York",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_2",
				"name": "Jane Smith",
				"experience": 8,
				"department": "Marketing",
				"role": "Marketing Manager",
				"skills": "Digital marketing, SEO, content strategy, analytics, social media",
				"location": "Los Angeles",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_3",
				"name": "Alice Johnson",
				"experience": 3,
				"department": "HR",
				"role": "HR Coordinator",
				"skills": "Recruitment, employee relations, HR policies, training programs",
				"location": "Chicago",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_4",
				"name": "Michael Brown",
				"experience": 12,
				"department": "Engineering",
				"role": "Senior Software Engineer",
				"skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
				"location": "San Francisco",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_5",
				"name": "Emily Wilson",
				"experience": 2,
				"department": "Marketing",
				"role": "Marketing Assistant",
				"skills": "Content creation, email marketing, market research, social media management",
				"location": "Austin",
				"employment_type": "Part-time"
			},
			{
				"id": "employee_6",
				"name": "David Lee",
				"experience": 15,
				"department": "Engineering",
				"role": "Engineering Manager",
				"skills": "Team leadership, project management, software architecture, mentoring",
				"location": "Seattle",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_7",
				"name": "Sarah Clark",
				"experience": 8,
				"department": "HR",
				"role": "HR Manager",
				"skills": "Performance management, compensation planning, policy development, conflict resolution",
				"location": "Boston",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_8",
				"name": "Chris Evans",
				"experience": 20,
				"department": "Engineering",
				"role": "Senior Architect",
				"skills": "System design, distributed systems, cloud platforms, technical strategy",
				"location": "New York",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_9",
				"name": "Jessica Taylor",
				"experience": 4,
				"department": "Marketing",
				"role": "Marketing Specialist",
				"skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
				"location": "Miami",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_10",
				"name": "Alex Rodriguez",
				"experience": 18,
				"department": "Engineering",
				"role": "Lead Software Engineer",
				"skills": "Full-stack development, React, Python, machine learning, data science",
				"location": "Denver",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_11",
				"name": "Hannah White",
				"experience": 6,
				"department": "HR",
				"role": "HR Business Partner",
				"skills": "Strategic HR, organizational development, change management, employee engagement",
				"location": "Portland",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_12",
				"name": "Kevin Martinez",
				"experience": 10,
				"department": "Engineering",
				"role": "DevOps Engineer",
				"skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
				"location": "Phoenix",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_13",
				"name": "Rachel Brown",
				"experience": 7,
				"department": "Marketing",
				"role": "Marketing Director",
				"skills": "Strategic marketing, team leadership, budget management, campaign optimization",
				"location": "Atlanta",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_14",
				"name": "Matthew Garcia",
				"experience": 3,
				"department": "Engineering",
				"role": "Junior Software Engineer",
				"skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
				"location": "Dallas",
				"employment_type": "Full-time"
			},
			{
				"id": "employee_15",
				"name": "Olivia Moore",
				"experience": 12,
				"department": "Engineering",
				"role": "Principal Engineer",
				"skills": "Technical leadership, system architecture, performance optimization, mentoring",
				"location": "San Francisco",
				"employment_type": "Full-time"
			},
		]
        # Create comprehensive text documents for each employee
        # These documents will be used for similarity search based on skills, roles, and experience
        employee_documents = []
        for employee in employees:
            document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
            document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
            document += f"Employment type: {employee['employment_type']}."
            employee_documents.append(document)
        
        #creating IDs
        ids = [employee['id'] for employee in employees]
        collection.upsert(
            documents=employee_documents,
            ids=ids,
            metadatas=[{
                "name": employee["name"],
                "department": employee["department"],
                "role": employee["role"],
                "experience": employee["experience"],
                "location": employee["location"],
                "employment_type": employee["employment_type"]
            } for employee in employees]
        )

        # Retrieving all items from the specified collection
        # The 'get' method fetches all records stored in the collection
        all_items = collection.get()
        # Logging the retrieved items to the console for inspection or debugging
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")
        #print(f"{all_items['documents'][0]}")
        perform_advanced_search(collection)

    except Exception as error:
        print(f"Error :{error}")

if __name__=="__main__":
    main()