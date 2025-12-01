# MCP Servers

This directory contains Model Context Protocol (MCP) servers for the generic agent application.

## Available Servers

### 1. SDR Server (`sdr_server/`)
The sales-focused MCP server with custom functionality for prospect research, email generation, campaign tracking, follow-up scheduling, and company insights.

### 2. HubSpot Server (`hubspot_server/`)
Custom HubSpot MCP server for CRM contact creation.

**Configuration:**
- Set `HUBSPOT_API_KEY` environment variable with your HubSpot Private App token

**Features:**
- **Create Contacts**: Create new contacts in HubSpot CRM with all standard fields
- **Fetch Contacts**: Search and retrieve contacts with optional filters:
  - General search query (searches across name, email, company)
  - Filter by company name
  - Filter by job title
  - Filter by lifecycle stage
  - Filter by email domain
  - Configurable result limits (1-100 contacts)
- **Update Contacts**: Update existing contacts with any combination of fields:
  - Standard contact fields (name, email, company, job title, phone, website)
  - Lifecycle stage and lead status management
  - Notes and additional contact information
- Supports all standard contact fields and HubSpot contact properties
- Automatic lifecycle stage assignment for new contacts
- Duplicate detection and error handling
- Comprehensive error reporting, validation, and pagination

## Running with Docker

To start all MCP servers:

```bash
cd SDR
docker-compose up --build
```

To start only the HubSpot MCP server:

```bash
docker-compose up --build hubspot-mcp-server
```

## Environment Variables

The `HUBSPOT_API_KEY` is configured directly in the `docker-compose.yml` file. For production use, you should:

1. Create a `.env` file in the project directory with:
```bash
HUBSPOT_API_KEY=your-hubspot-api-key-here
```

2. Update `docker-compose.yml` to use:
```yaml
- HUBSPOT_API_KEY=${HUBSPOT_API_KEY}
```

## API Access

- Generic Agent Flask App: http://localhost:8080
- MCP servers communicate via stdio (not HTTP)

## Getting HubSpot API Key

1. Go to your HubSpot account
2. Navigate to Settings → Integrations → Private Apps
3. Create a new private app or use an existing one
4. Grant the necessary scopes:
   - `crm.objects.contacts.read` (for fetching contacts)
   - `crm.objects.contacts.write` (for creating and updating contacts)
5. Copy the access token and set it as `HUBSPOT_API_KEY` 