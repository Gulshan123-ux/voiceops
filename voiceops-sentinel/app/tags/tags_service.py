from __future__ import annotations
from app.database import get_call_by_id, update_call_tags

def add_tag(job_id: str, tag: str) -> list[str]:
    """Add a tag to a call record and return the updated tags list."""
    tag = tag.strip().lower()
    if not tag:
        return []
    
    call_record = get_call_by_id(job_id)
    if not call_record:
        raise ValueError(f"Call record with ID '{job_id}' not found.")
    
    tags = call_record.get("tags") or []
    if tag not in tags:
        tags.append(tag)
        update_call_tags(job_id, tags)
    
    return tags

def remove_tag(job_id: str, tag: str) -> list[str]:
    """Remove a tag from a call record and return the updated tags list."""
    tag = tag.strip().lower()
    call_record = get_call_by_id(job_id)
    if not call_record:
        raise ValueError(f"Call record with ID '{job_id}' not found.")
    
    tags = call_record.get("tags") or []
    if tag in tags:
        tags.remove(tag)
        update_call_tags(job_id, tags)
    
    return tags
