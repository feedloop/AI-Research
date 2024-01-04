from utils.api import get_embeddings_ada


def get_resource_id(cursor, resource_name):
    query = """SELECT * FROM resource WHERE name = %s;"""
    cursor.execute(query, (resource_name,))
    result = cursor.fetchone()

    return result[0]


def get_retrieved_knowledge(cursor, question, resource_ids, top_k):
    query = """
    SELECT 
        c.id,
        json_build_object('context', c.context, 'fact', c.fact, 'number', c.number, 'summary', c.summary, 'filetype', m.filetype, 'name', r.name, 'project', r.project_id)::jsonb AS metadata,
        1 - (c.embeddings <=> %(query_embedding)s::vector) AS similarity
    FROM fact c JOIN resource r ON c.resource_id = r.id JOIN metadata m ON r.id = m.resource_id
    WHERE c.resource_id = ANY (%(resource_ids)s::UUID[])
    ORDER BY c.embeddings <=> %(query_embedding)s::vector
    LIMIT %(match_count)s;
    """
    cursor.execute(
        query,
        {
            "query_embedding": get_embeddings_ada(question),
            "resource_ids": resource_ids,
            "match_count": top_k,
        },
    )
    result = cursor.fetchall()

    return result


def delete_facts_resource(conn, cursor, resource_id):
    delete_query = """
    DELETE FROM fact WHERE resource_id = %s;
    """
    cursor.execute(delete_query, (resource_id,))
    conn.commit()


def insert_fact_resource(conn, cursor, data):
    insert_query = """
    INSERT INTO fact (context, fact, resource_id, embeddings, summary, number)
    VALUES (%(context)s, %(fact)s, %(resource_id)s, %(embeddings)s, %(summary)s, %(number)s);
    """
    cursor.execute(insert_query, data)
    conn.commit()
