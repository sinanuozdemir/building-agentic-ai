#!/usr/bin/env python3
"""
HubSpot MCP Server for contact creation
This server provides a single tool to create contacts in HubSpot CRM.
"""

import asyncio
import json
import os
import httpx
from typing import Any, Dict, List
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types

# Create server instance
server = Server("hubspot-mcp-server")

# HubSpot API configuration
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
HUBSPOT_BASE_URL = "https://api.hubapi.com"

if not HUBSPOT_API_KEY:
    raise ValueError("HUBSPOT_API_KEY environment variable is required")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools for HubSpot contact management."""
    return [
        Tool(
            name="create_contact",
            description="Create a new contact in HubSpot CRM",
            inputSchema={
                "type": "object",
                "properties": {
                    "first_name": {
                        "type": "string",
                        "description": "First name of the contact"
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Last name of the contact"
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address of the contact"
                    },
                    "company": {
                        "type": "string",
                        "description": "Company name where the contact works"
                    },
                    "job_title": {
                        "type": "string",
                        "description": "Job title of the contact"
                    },
                    "phone": {
                        "type": "string",
                        "description": "Phone number of the contact"
                    },
                    "website": {
                        "type": "string",
                        "description": "Company website URL"
                    }
                },
                "required": ["email"]
            }
        ),
        Tool(
            name="fetch_contacts",
            description="Fetch contacts from HubSpot CRM with optional filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "General search query to find contacts (searches across name, email, company)"
                    },
                    "company": {
                        "type": "string",
                        "description": "Filter by company name"
                    },
                    "job_title": {
                        "type": "string",
                        "description": "Filter by job title"
                    },
                    "email_domain": {
                        "type": "string",
                        "description": "Filter by email domain (e.g., 'gmail.com', 'company.com')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of contacts to return (default: 10, max: 100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="update_contact",
            description="Update an existing contact in HubSpot CRM",
            inputSchema={
                "type": "object",
                "properties": {
                    "contact_id": {
                        "type": "string",
                        "description": "HubSpot contact ID to update"
                    },
                    "first_name": {
                        "type": "string",
                        "description": "Updated first name of the contact"
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Updated last name of the contact"
                    },
                    "email": {
                        "type": "string",
                        "description": "Updated email address of the contact"
                    },
                    "company": {
                        "type": "string",
                        "description": "Updated company name where the contact works"
                    },
                    "job_title": {
                        "type": "string",
                        "description": "Updated job title of the contact"
                    },
                    "phone": {
                        "type": "string",
                        "description": "Updated phone number of the contact"
                    },
                    "website": {
                        "type": "string",
                        "description": "Updated company website URL"
                    },
                    "lead_status": {
                        "type": "string",
                        "enum": ["NEW", "OPEN", "IN_PROGRESS", "OPEN_DEAL", "UNQUALIFIED", "ATTEMPTED_TO_CONTACT", "CONNECTED", "BAD_TIMING"],
                        "description": "Updated lead status"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes about the contact"
                    }
                },
                "required": ["contact_id"]
            }
        ),
        Tool(
            name="attach_note_to_contact",
            description="Create a new note and attach it to a specific contact in HubSpot CRM",
            inputSchema={
                "type": "object",
                "properties": {
                    "contact_id": {
                        "type": "string",
                        "description": "HubSpot contact ID to attach the note to"
                    },
                    "note_body": {
                        "type": "string",
                        "description": "The content/body of the note"
                    },
                    "note_title": {
                        "type": "string",
                        "description": "Optional title for the note"
                    }
                },
                "required": ["contact_id", "note_body"]
            }
        ),
        Tool(
            name="retrieve_all_notes_for_contact",
            description="Retrieve all notes associated with a specific contact from HubSpot CRM",
            inputSchema={
                "type": "object",
                "properties": {
                    "contact_id": {
                        "type": "string",
                        "description": "HubSpot contact ID to retrieve notes for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of notes to return (default: 20, max: 100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20
                    }
                },
                "required": ["contact_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls for HubSpot contact management."""
    
    if name == "create_contact":
        try:
            # Extract contact information from arguments
            email = arguments.get("email")
            first_name = arguments.get("first_name", "")
            last_name = arguments.get("last_name", "")
            company = arguments.get("company", "")
            job_title = arguments.get("job_title", "")
            phone = arguments.get("phone", "")
            website = arguments.get("website", "")
            
            # Prepare the contact data for HubSpot API
            contact_data = {
                "properties": {
                    "email": email
                }
            }
            
            # Add optional fields if provided
            if first_name:
                contact_data["properties"]["firstname"] = first_name
            if last_name:
                contact_data["properties"]["lastname"] = last_name
            if company:
                contact_data["properties"]["company"] = company
            if job_title:
                contact_data["properties"]["jobtitle"] = job_title
            if phone:
                contact_data["properties"]["phone"] = phone
            if website:
                contact_data["properties"]["website"] = website
            
            # Make API call to HubSpot
            headers = {
                "Authorization": f"Bearer {HUBSPOT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts",
                    headers=headers,
                    json=contact_data,
                    timeout=30.0
                )
                
                if response.status_code == 201:
                    # Success - contact created
                    contact_info = response.json()
                    contact_id = contact_info.get("id")
                    
                    result = f"✅ Contact created successfully in HubSpot!\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += f"Email: {email}\n"
                    
                    if first_name or last_name:
                        full_name = f"{first_name} {last_name}".strip()
                        result += f"Name: {full_name}\n"
                    
                    if company:
                        result += f"Company: {company}\n"
                    
                    if job_title:
                        result += f"Job Title: {job_title}\n"
                    
                    if phone:
                        result += f"Phone: {phone}\n"
                    
                    if website:
                        result += f"Website: {website}\n"
                    
                    result += f"\nContact URL: https://app.hubspot.com/contacts/[portal-id]/contact/{contact_id}"
                    
                    return [types.TextContent(type="text", text=result)]
                
                elif response.status_code == 409:
                    # Contact already exists
                    error_details = response.json()
                    result = f"⚠️ Contact already exists in HubSpot\n\n"
                    result += f"Email: {email}\n"
                    result += f"Error: {error_details.get('message', 'Contact with this email already exists')}\n\n"
                    result += "You may want to update the existing contact instead of creating a new one."
                    
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    # Other error
                    error_details = response.json() if response.content else {}
                    error_message = error_details.get('message', f'HTTP {response.status_code}')
                    
                    result = f"❌ Failed to create contact in HubSpot\n\n"
                    result += f"Status Code: {response.status_code}\n"
                    result += f"Error: {error_message}\n"
                    result += f"Email: {email}"
                    
                    return [types.TextContent(type="text", text=result)]
        
        except httpx.TimeoutException:
            result = f"❌ Request timed out while creating contact in HubSpot\n\n"
            result += f"Email: {email}\n"
            result += "Please try again or check your network connection."
            return [types.TextContent(type="text", text=result)]
        
        except httpx.RequestError as e:
            result = f"❌ Network error while creating contact in HubSpot\n\n"
            result += f"Email: {email}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
        
        except Exception as e:
            result = f"❌ Unexpected error while creating contact in HubSpot\n\n"
            result += f"Email: {email}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
    
    elif name == "fetch_contacts":
        try:
            # Extract filter parameters
            search_query = arguments.get("search_query", "")
            company = arguments.get("company", "")
            job_title = arguments.get("job_title", "")
            email_domain = arguments.get("email_domain", "")
            lead_status = arguments.get("lead_status", "")
            limit = arguments.get("limit", 10)
            
            # Build search filters
            filters = []
            filter_groups = []
            
            # Add company filter
            if company:
                filters.append({
                    "propertyName": "company",
                    "operator": "CONTAINS_TOKEN",
                    "value": company
                })
            
            # Add job title filter
            if job_title:
                filters.append({
                    "propertyName": "jobtitle",
                    "operator": "CONTAINS_TOKEN",
                    "value": job_title
                })
            
            # Add email domain filter
            if email_domain:
                filters.append({
                    "propertyName": "email",
                    "operator": "CONTAINS_TOKEN",
                    "value": email_domain
                })

            # Add lead status filter
            if lead_status:
                filters.append({
                    "propertyName": "hs_lead_status",
                    "operator": "EQ",
                    "value": lead_status
                })
            
            # Build search payload
            search_payload = {
                "limit": min(limit, 100),  # Ensure we don't exceed API limits
                "properties": [
                    "firstname", "lastname", "email", "company", 
                    "jobtitle", "phone", "website",
                    "createdate", "lastmodifieddate", "hs_lead_status"
                ]
            }
            
            # Add filters if any exist
            if filters:
                filter_groups.append({
                    "filters": filters
                })
                search_payload["filterGroups"] = filter_groups
            
            # Add general search query if provided
            if search_query:
                search_payload["query"] = search_query
            
            # Make API call to HubSpot
            headers = {
                "Authorization": f"Bearer {HUBSPOT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                # Use search endpoint for better filtering capabilities
                response = await client.post(
                    f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/search",
                    headers=headers,
                    json=search_payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    # Success - contacts retrieved
                    data = response.json()
                    contacts = data.get("results", [])
                    total = data.get("total", 0)
                    
                    if not contacts:
                        result = "📭 No contacts found matching the specified criteria.\n\n"
                        if any([search_query, company, job_title, email_domain, lead_status]):
                            result += "Applied filters:\n"
                            if search_query:
                                result += f"• Search Query: {search_query}\n"
                            if company:
                                result += f"• Company: {company}\n"
                            if job_title:
                                result += f"• Job Title: {job_title}\n"
                            if email_domain:
                                result += f"• Email Domain: {email_domain}\n"
                            if lead_status:
                                result += f"• Lead Status: {lead_status}\n"
                        else:
                            result += "No filters applied - your HubSpot may not have any contacts yet."
                        
                        return [types.TextContent(type="text", text=result)]
                    
                    result = f"📋 Found {len(contacts)} contacts"
                    if total > len(contacts):
                        result += f" (showing first {len(contacts)} of {total} total)"
                    result += "\n\n"
                    
                    # Show applied filters
                    if any([search_query, company, job_title, email_domain, lead_status]):
                        result += "Applied filters:\n"
                        if search_query:
                            result += f"• Search Query: {search_query}\n"
                        if company:
                            result += f"• Company: {company}\n"
                        if job_title:
                            result += f"• Job Title: {job_title}\n"
                        if email_domain:
                            result += f"• Email Domain: {email_domain}\n"
                        if lead_status:
                            result += f"• Lead Status: {lead_status}\n"
                        result += "\n"
                    
                    # Display contact details
                    for i, contact in enumerate(contacts, 1):
                        props = contact.get("properties", {})
                        contact_id = contact.get("id")
                        
                        result += f"{i}. "
                        
                        # Name
                        first_name = props.get("firstname", "")
                        last_name = props.get("lastname", "")
                        if first_name or last_name:
                            full_name = f"{first_name} {last_name}".strip()
                            result += f"{full_name}"
                        else:
                            result += "No Name"
                        
                        # Email
                        email = props.get("email", "")
                        if email:
                            result += f" ({email})"
                        
                        result += f"\n   ID: {contact_id}\n"
                        
                        # Company
                        company_name = props.get("company", "")
                        if company_name:
                            result += f"   Company: {company_name}\n"
                        
                        # Job title
                        title = props.get("jobtitle", "")
                        if title:
                            result += f"   Job Title: {title}\n"
                        
                        # Lead status
                        lead_status_val = props.get("hs_lead_status", "")
                        if lead_status_val:
                            result += f"   Lead Status: {lead_status_val}\n"
                        
                        # Phone
                        phone = props.get("phone", "")
                        if phone:
                            result += f"   Phone: {phone}\n"
                        
                        # Website
                        website = props.get("website", "")
                        if website:
                            result += f"   Website: {website}\n"
                        
                        # Creation date
                        create_date = props.get("createdate", "")
                        if create_date:
                            # Convert timestamp to readable date
                            from datetime import datetime
                            try:
                                dt = datetime.fromisoformat(create_date.replace('Z', '+00:00'))
                                result += f"   Created: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                            except:
                                result += f"   Created: {create_date}\n"
                        
                        result += f"   HubSpot URL: https://app.hubspot.com/contacts/[portal-id]/contact/{contact_id}\n\n"
                    
                    # Add pagination info if there are more results
                    if total > len(contacts):
                        result += f"💡 Showing first {len(contacts)} results. "
                        result += f"There are {total - len(contacts)} more contacts matching your criteria."
                    
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    # Error response
                    error_details = response.json() if response.content else {}
                    error_message = error_details.get('message', f'HTTP {response.status_code}')
                    
                    result = f"❌ Failed to fetch contacts from HubSpot\n\n"
                    result += f"Status Code: {response.status_code}\n"
                    result += f"Error: {error_message}"
                    
                    return [types.TextContent(type="text", text=result)]
        
        except httpx.TimeoutException:
            result = f"❌ Request timed out while fetching contacts from HubSpot\n\n"
            result += "Please try again or check your network connection."
            return [types.TextContent(type="text", text=result)]
        
        except httpx.RequestError as e:
            result = f"❌ Network error while fetching contacts from HubSpot\n\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
        
        except Exception as e:
            result = f"❌ Unexpected error while fetching contacts from HubSpot\n\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
    
    elif name == "update_contact":
        try:
            # Extract contact ID and update fields
            contact_id = arguments.get("contact_id")
            if not contact_id:
                result = "❌ Contact ID is required for updating a contact"
                return [types.TextContent(type="text", text=result)]
            
            # Extract optional update fields
            first_name = arguments.get("first_name")
            last_name = arguments.get("last_name")
            email = arguments.get("email")
            company = arguments.get("company")
            job_title = arguments.get("job_title")
            phone = arguments.get("phone")
            website = arguments.get("website")
            lead_status = arguments.get("lead_status")
            notes = arguments.get("notes")
            
            # Build properties object with only provided fields
            properties = {}
            
            if first_name is not None:
                properties["firstname"] = first_name
            if last_name is not None:
                properties["lastname"] = last_name
            if email is not None:
                properties["email"] = email
            if company is not None:
                properties["company"] = company
            if job_title is not None:
                properties["jobtitle"] = job_title
            if phone is not None:
                properties["phone"] = phone
            if website is not None:
                properties["website"] = website
            # Note: Lead status property may not exist in all HubSpot instances
            # Try to add it, but the API call may fail if the property doesn't exist
            if lead_status is not None:
                properties["hs_lead_status"] = lead_status
            if notes is not None:
                properties["notes"] = notes
            
            if not properties:
                result = "❌ At least one property must be provided to update the contact"
                return [types.TextContent(type="text", text=result)]
            
            # Prepare the update data
            update_data = {
                "properties": properties
            }
            
            # Make API call to HubSpot
            headers = {
                "Authorization": f"Bearer {HUBSPOT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/{contact_id}",
                    headers=headers,
                    json=update_data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    # Success - contact updated
                    contact_info = response.json()
                    updated_properties = contact_info.get("properties", {})
                    
                    result = f"✅ Contact updated successfully in HubSpot!\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    
                    # Show updated fields
                    result += "Updated fields:\n"
                    for prop_key, prop_value in properties.items():
                        # Convert back to readable names
                        readable_names = {
                            "firstname": "First Name",
                            "lastname": "Last Name", 
                            "email": "Email",
                            "company": "Company",
                            "jobtitle": "Job Title",
                            "phone": "Phone",
                            "website": "Website",
                            "hs_lead_status": "Lead Status",
                            "notes": "Notes"
                        }
                        field_name = readable_names.get(prop_key, prop_key)
                        result += f"• {field_name}: {prop_value}\n"
                    
                    # Show current contact details
                    result += f"\nCurrent contact details:\n"
                    
                    # Name
                    first = updated_properties.get("firstname", "")
                    last = updated_properties.get("lastname", "")
                    if first or last:
                        full_name = f"{first} {last}".strip()
                        result += f"Name: {full_name}\n"
                    
                    # Email
                    current_email = updated_properties.get("email", "")
                    if current_email:
                        result += f"Email: {current_email}\n"
                    
                    # Company
                    current_company = updated_properties.get("company", "")
                    if current_company:
                        result += f"Company: {current_company}\n"
                    
                    # Job Title
                    current_title = updated_properties.get("jobtitle", "")
                    if current_title:
                        result += f"Job Title: {current_title}\n"
                    
                    # Phone
                    current_phone = updated_properties.get("phone", "")
                    if current_phone:
                        result += f"Phone: {current_phone}\n"
                    
                    # Website
                    current_website = updated_properties.get("website", "")
                    if current_website:
                        result += f"Website: {current_website}\n"
                    
                    # Lead Status
                    current_lead_status = updated_properties.get("hs_lead_status", "")
                    if current_lead_status:
                        result += f"Lead Status: {current_lead_status}\n"
                    
                    # Last Modified
                    last_modified = updated_properties.get("lastmodifieddate", "")
                    if last_modified:
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                            result += f"Last Modified: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                        except:
                            result += f"Last Modified: {last_modified}\n"
                    
                    result += f"\nContact URL: https://app.hubspot.com/contacts/[portal-id]/contact/{contact_id}"
                    
                    return [types.TextContent(type="text", text=result)]
                
                elif response.status_code == 404:
                    # Contact not found
                    result = f"❌ Contact not found in HubSpot\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += "Please verify the contact ID is correct and the contact exists."
                    
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    # Other error
                    error_details = response.json() if response.content else {}
                    error_message = error_details.get('message', f'HTTP {response.status_code}')
                    
                    result = f"❌ Failed to update contact in HubSpot\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += f"Status Code: {response.status_code}\n"
                    result += f"Error: {error_message}"
                    
                    return [types.TextContent(type="text", text=result)]
        
        except httpx.TimeoutException:
            result = f"❌ Request timed out while updating contact in HubSpot\n\n"
            result += f"Contact ID: {contact_id}\n"
            result += "Please try again or check your network connection."
            return [types.TextContent(type="text", text=result)]
        
        except httpx.RequestError as e:
            result = f"❌ Network error while updating contact in HubSpot\n\n"
            result += f"Contact ID: {contact_id}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
        
        except Exception as e:
            result = f"❌ Unexpected error while updating contact in HubSpot\n\n"
            result += f"Contact ID: {contact_id if 'contact_id' in locals() else 'unknown'}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
    
    elif name == "attach_note_to_contact":
        try:
            # Extract contact ID and note details
            contact_id = arguments.get("contact_id")
            if not contact_id:
                result = "❌ Contact ID is required for attaching a note"
                return [types.TextContent(type="text", text=result)]
            
            note_body = arguments.get("note_body")
            if not note_body:
                result = "❌ Note body is required for attaching a note"
                return [types.TextContent(type="text", text=result)]
            
            note_title = arguments.get("note_title", "")
            
            # Prepare the note data with proper HubSpot properties and associations
            import time
            current_timestamp = int(time.time() * 1000)  # HubSpot expects milliseconds
            
            note_data = {
                "properties": {
                    "hs_note_body": note_body,
                    "hs_timestamp": str(current_timestamp)
                },
                "associations": [
                    {
                        "to": {
                            "id": contact_id
                        },
                        "types": [
                            {
                                "associationCategory": "HUBSPOT_DEFINED",
                                "associationTypeId": 202  # Contact to Note association
                            }
                        ]
                    }
                ]
            }
            
            # Add title if provided
            if note_title:
                note_data["properties"]["hs_note_title"] = note_title
            
            # Make API call to HubSpot
            headers = {
                "Authorization": f"Bearer {HUBSPOT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{HUBSPOT_BASE_URL}/crm/v3/objects/notes",
                    headers=headers,
                    json=note_data,
                    timeout=30.0
                )
                
                if response.status_code == 201:
                    # Success - note created and attached
                    note_info = response.json()
                    note_id = note_info.get("id")
                    
                    result = f"✅ Note attached successfully to contact in HubSpot!\n\n"
                    result += f"Note ID: {note_id}\n"
                    result += f"Contact ID: {contact_id}\n"
                    if note_title:
                        result += f"Note Title: {note_title}\n"
                    result += f"Note Body: {note_body}\n"
                    result += f"\nContact URL: https://app.hubspot.com/contacts/[portal-id]/contact/{contact_id}"
                    
                    return [types.TextContent(type="text", text=result)]
                
                elif response.status_code == 404:
                    # Contact not found or invalid association
                    result = f"❌ Contact not found or invalid association in HubSpot\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += "Please verify the contact ID is correct and the contact exists."
                    
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    # Other error
                    error_details = response.json() if response.content else {}
                    error_message = error_details.get('message', f'HTTP {response.status_code}')
                    
                    result = f"❌ Failed to attach note to contact in HubSpot\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += f"Status Code: {response.status_code}\n"
                    result += f"Error: {error_message}"
                    
                    return [types.TextContent(type="text", text=result)]
        
        except httpx.TimeoutException:
            result = f"❌ Request timed out while attaching note to contact in HubSpot\n\n"
            result += f"Contact ID: {contact_id}\n"
            result += "Please try again or check your network connection."
            return [types.TextContent(type="text", text=result)]
        
        except httpx.RequestError as e:
            result = f"❌ Network error while attaching note to contact in HubSpot\n\n"
            result += f"Contact ID: {contact_id}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
        
        except Exception as e:
            result = f"❌ Unexpected error while attaching note to contact in HubSpot\n\n"
            result += f"Contact ID: {contact_id if 'contact_id' in locals() else 'unknown'}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
    
    elif name == "retrieve_all_notes_for_contact":
        try:
            # Extract contact ID and limit
            contact_id = arguments.get("contact_id")
            if not contact_id:
                result = "❌ Contact ID is required for retrieving notes"
                return [types.TextContent(type="text", text=result)]
            
            limit = arguments.get("limit", 20)
            
            # Step 1: Get all notes associated with the contact using associations API
            headers = {
                "Authorization": f"Bearer {HUBSPOT_API_KEY}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                # Get associated notes for the contact
                response = await client.get(
                    f"{HUBSPOT_BASE_URL}/crm/v4/objects/contacts/{contact_id}/associations/notes",
                    headers=headers,
                    params={"limit": limit},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    associations_data = response.json()
                    associations = associations_data.get("results", [])
                    
                    if not associations:
                        result = f"📭 No notes found for contact {contact_id}.\n\n"
                        result += "This contact has no associated notes."
                        
                        return [types.TextContent(type="text", text=result)]
                    
                    # Extract note IDs from associations
                    note_ids = [assoc.get("toObjectId") for assoc in associations if assoc.get("toObjectId")]
                    
                    if not note_ids:
                        result = f"📭 No valid note associations found for contact {contact_id}.\n\n"
                        return [types.TextContent(type="text", text=result)]
                    
                    # Step 2: Batch read the note details
                    batch_read_payload = {
                        "properties": [
                            "hs_note_body", "hs_note_title", "hs_timestamp", 
                            "createdate", "lastmodifieddate"
                        ],
                        "inputs": [{"id": note_id} for note_id in note_ids[:limit]]
                    }
                    
                    batch_response = await client.post(
                        f"{HUBSPOT_BASE_URL}/crm/v3/objects/notes/batch/read",
                        headers=headers,
                        json=batch_read_payload,
                        timeout=30.0
                    )
                    
                    if batch_response.status_code == 200:
                        # Success - notes retrieved
                        batch_data = batch_response.json()
                        notes = batch_data.get("results", [])
                        
                        result = f"📋 Found {len(notes)} notes for contact {contact_id}\n\n"
                        
                        # Display note details
                        for i, note in enumerate(notes, 1):
                            props = note.get("properties", {})
                            note_id = note.get("id")
                            
                            result += f"{i}. Note ID: {note_id}\n"
                            
                            # Title
                            title = props.get("hs_note_title", "")
                            if title:
                                result += f"   Title: {title}\n"
                            
                            # Body
                            body = props.get("hs_note_body", "")
                            if body:
                                # Truncate long notes for display
                                display_body = body[:200] + "..." if len(body) > 200 else body
                                result += f"   Body: {display_body}\n"
                            
                            # Timestamp (when the note was actually made, not just created in system)
                            timestamp = props.get("hs_timestamp", "")
                            if timestamp:
                                from datetime import datetime
                                try:
                                    # Convert milliseconds to seconds
                                    dt = datetime.fromtimestamp(int(timestamp) / 1000)
                                    result += f"   Note Date: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                                except:
                                    result += f"   Note Date: {timestamp}\n"
                            
                            # Creation date
                            create_date = props.get("createdate", "")
                            if create_date:
                                # Convert timestamp to readable date
                                from datetime import datetime
                                try:
                                    dt = datetime.fromisoformat(create_date.replace('Z', '+00:00'))
                                    result += f"   Created: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                                except:
                                    result += f"   Created: {create_date}\n"
                            
                            # Last modified date
                            modified_date = props.get("lastmodifieddate", "")
                            if modified_date and modified_date != create_date:
                                from datetime import datetime
                                try:
                                    dt = datetime.fromisoformat(modified_date.replace('Z', '+00:00'))
                                    result += f"   Last Modified: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                                except:
                                    result += f"   Last Modified: {modified_date}\n"
                            
                            result += f"   HubSpot URL: https://app.hubspot.com/contacts/[portal-id]/contact/{contact_id}\n\n"
                        
                        # Add pagination info if there are more results
                        total_associations = associations_data.get("total", len(associations))
                        if total_associations > len(notes):
                            result += f"💡 Showing first {len(notes)} results. "
                            result += f"There are {total_associations - len(notes)} more notes for this contact."
                        
                        return [types.TextContent(type="text", text=result)]
                    
                    else:
                        # Error in batch read
                        error_details = batch_response.json() if batch_response.content else {}
                        error_message = error_details.get('message', f'HTTP {batch_response.status_code}')
                        
                        result = f"❌ Failed to retrieve note details from HubSpot\n\n"
                        result += f"Contact ID: {contact_id}\n"
                        result += f"Status Code: {batch_response.status_code}\n"
                        result += f"Error: {error_message}"
                        
                        return [types.TextContent(type="text", text=result)]
                
                elif response.status_code == 404:
                    # Contact not found
                    result = f"❌ Contact not found in HubSpot\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += "Please verify the contact ID is correct and the contact exists."
                    
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    # Error response
                    error_details = response.json() if response.content else {}
                    error_message = error_details.get('message', f'HTTP {response.status_code}')
                    
                    result = f"❌ Failed to retrieve note associations from HubSpot\n\n"
                    result += f"Contact ID: {contact_id}\n"
                    result += f"Status Code: {response.status_code}\n"
                    result += f"Error: {error_message}"
                    
                    return [types.TextContent(type="text", text=result)]
        
        except httpx.TimeoutException:
            result = f"❌ Request timed out while retrieving notes from HubSpot\n\n"
            result += f"Contact ID: {contact_id}\n"
            result += "Please try again or check your network connection."
            return [types.TextContent(type="text", text=result)]
        
        except httpx.RequestError as e:
            result = f"❌ Network error while retrieving notes from HubSpot\n\n"
            result += f"Contact ID: {contact_id}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
        
        except Exception as e:
            result = f"❌ Unexpected error while retrieving notes from HubSpot\n\n"
            result += f"Contact ID: {contact_id if 'contact_id' in locals() else 'unknown'}\n"
            result += f"Error: {str(e)}"
            return [types.TextContent(type="text", text=result)]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main server entry point."""
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="hubspot-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import mcp.server.stdio
    asyncio.run(main()) 