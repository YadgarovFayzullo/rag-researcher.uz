from core.clients import get_qdrant
from core.config import settings

if __name__ == "__main__":
    get_qdrant().delete_collection(settings.QDRANT_COLLECTION)
    print("Удалено")
