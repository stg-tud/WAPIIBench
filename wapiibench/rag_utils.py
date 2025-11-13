from __future__ import annotations

import math
import re
from collections import Counter
from operator import itemgetter

import numpy as np
import yaml


class Retriever:

    def __init__(self, spec_file: str) -> None:
        """
        Create a retriever for OpenAPI specifications that can be used for retrieval-augmented generation (RAG).
        :param spec_file: Path to the specification
        """
        # Load YAML spec directly
        with open(spec_file, 'r') as file:
            self.spec = yaml.safe_load(file)

        # Validate basic structure
        assert isinstance(self.spec, dict), "Invalid OpenAPI specification format"

        # Extract all endpoints with their descriptive text and cache them
        self.endpoints = self._extract_all_endpoints(self.spec)

    def retrieve_spec_for_task(self, task: str, num_chunks: int = 5, truncation_threshold: int | None = None,
                               return_dict: bool = False) -> str | dict:
        """
        Retrieve chunks (endpoints) from the specification that might be relevant for the given task.
        :param task: The task
        :param num_chunks: Number of chunks to retrieve
        :param truncation_threshold: Maximum number of characters in the retrieved string
        :param return_dict: If True, return the retrieval results in a dict, not serialized as a string
        :return: YAML string with structured endpoint information
        """
        # Score each endpoint based on CF-IDF similarity with the task
        scored_endpoints = self._calculate_cfidf_similarity(task)

        # Sort by score (descending) and take top N
        scored_endpoints.sort(key=itemgetter(0), reverse=True)
        top_endpoints = scored_endpoints[:num_chunks]

        # Generate structured documentation for each selected endpoint
        endpoint_dicts = []
        for score, endpoint in top_endpoints:
            endpoint_dict = self._format_endpoint_dict(endpoint)
            endpoint_dicts.append(endpoint_dict)

        result = {
            "title": self.spec['info']['title'],
            "server_url": self.spec['servers'][0]['url'],
            "paths": endpoint_dicts,
        }

        if return_dict:
            return result

        # Convert YAML to string
        result_str = yaml.safe_dump(result, indent=2, width=float('inf'), allow_unicode=True, sort_keys=False)

        if truncation_threshold is not None and len(result_str) > truncation_threshold:
            # Very simple truncation strategy; this could be improved
            result_str = result_str[:truncation_threshold]

        return result_str

    def _resolve_ref(self, ref_path: str) -> dict[str, any]:
        """
        Resolve a $ref reference to its actual definition.
        :param ref_path: The $ref path (e.g., "#/components/parameters/project_path_gid")
        :return: The resolved definition
        """
        if not ref_path.startswith('#/'):
            return {}

        # Remove the #/ prefix and split the path
        path_parts = ref_path[2:].split('/')
        current = self.spec

        # Navigate through the spec to find the referenced definition
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {}

        return current if isinstance(current, dict) else {}

    def _resolve_parameter(self, param: dict[str, any]) -> dict[str, any]:
        """
        Resolve a parameter, handling $ref references.
        :param param: Parameter dictionary that might contain $ref
        :return: Resolved parameter dictionary
        """
        if not isinstance(param, dict):
            return param

        # If parameter has $ref, resolve it
        if '$ref' in param:
            ref_path = param['$ref']
            resolved_param = self._resolve_ref(ref_path)
            if resolved_param:
                return resolved_param

        return param

    def _resolve_schema(self, schema: dict[str, any], visited: set = None) -> dict[str, any]:
        """
        Resolve a schema, handling $ref references recursively.
        :param schema: Schema dictionary that might contain $ref
        :param visited: Set of visited schemas to prevent infinite recursion
        :return: Resolved schema dictionary
        """
        if visited is None:
            visited = set()

        if not isinstance(schema, dict):
            return schema

        # Prevent infinite recursion
        schema_id = id(schema)
        if schema_id in visited:
            return schema
        visited.add(schema_id)

        # If schema has $ref, resolve it
        if '$ref' in schema:
            ref_path = schema['$ref']
            resolved_schema = self._resolve_ref(ref_path)
            if resolved_schema:
                # Recursively resolve the referenced schema
                return self._resolve_schema(resolved_schema, visited)

        # Recursively resolve nested schemas
        resolved_schema = schema.copy()

        # Resolve properties
        if 'properties' in resolved_schema and isinstance(resolved_schema['properties'], dict):
            resolved_properties = {}
            for prop_name, prop_schema in resolved_schema['properties'].items():
                if isinstance(prop_schema, dict):
                    resolved_properties[prop_name] = self._resolve_schema(prop_schema, visited)
                else:
                    resolved_properties[prop_name] = prop_schema
            resolved_schema['properties'] = resolved_properties

        # Resolve items for arrays
        if 'items' in resolved_schema and isinstance(resolved_schema['items'], dict):
            resolved_schema['items'] = self._resolve_schema(resolved_schema['items'], visited)

        # Resolve allOf, oneOf, anyOf
        for key in ['allOf', 'oneOf', 'anyOf']:
            if key in resolved_schema and isinstance(resolved_schema[key], list):
                resolved_items = []
                for item in resolved_schema[key]:
                    if isinstance(item, dict):
                        resolved_items.append(self._resolve_schema(item, visited))
                    else:
                        resolved_items.append(item)
                resolved_schema[key] = resolved_items

        return resolved_schema

    def _extract_all_endpoints(self, spec: dict) -> list[dict[str, any]]:
        """
        Extract all endpoints from the YAML specification with their descriptive text.
        :param spec: YAML specification
        :return: List of endpoint dictionaries with text and metadata
        """
        endpoints = []

        # Get paths from spec
        paths = spec.get('paths', {})
        if not isinstance(paths, dict):
            return endpoints

        # Iterate through paths
        for path_url, path_data in paths.items():
            if not isinstance(path_data, dict):
                continue

            # Get path-level parameters and resolve $ref
            path_params = path_data.get('parameters', [])
            if not isinstance(path_params, list):
                path_params = []

            # Resolve $ref in path parameters
            resolved_path_params = []
            for param in path_params:
                resolved_param = self._resolve_parameter(param)
                if resolved_param:
                    resolved_path_params.append(resolved_param)

            # Process each HTTP method
            for method, operation_data in path_data.items():
                if method in ['parameters', '$ref'] or not isinstance(operation_data, dict):
                    continue

                # Gather descriptive text for this endpoint
                text_parts = []

                # Add HTTP method and path
                text_parts.append(f"{method.upper()} {path_url}")

                # Add summary if present
                summary = operation_data.get('summary', '')
                if summary:
                    text_parts.append(summary)

                # Add description if present
                description = operation_data.get('description', '')
                if description:
                    text_parts.append(description)

                # Add operation ID if present
                operation_id = operation_data.get('operationId', '')
                if operation_id:
                    text_parts.append(operation_id)

                # Get operation-level parameters and resolve $ref
                operation_params = operation_data.get('parameters', [])
                if not isinstance(operation_params, list):
                    operation_params = []

                # Resolve $ref in operation parameters
                resolved_operation_params = []
                for param in operation_params:
                    resolved_param = self._resolve_parameter(param)
                    if resolved_param:
                        resolved_operation_params.append(resolved_param)

                # Merge and deduplicate parameters by name
                all_params = []
                seen_param_names = set()

                # Add path-level parameters first
                for param in resolved_path_params:
                    if isinstance(param, dict) and 'name' in param:
                        param_name = param['name']
                        if param_name not in seen_param_names:
                            all_params.append(param)
                            seen_param_names.add(param_name)

                # Add operation-level parameters (override path-level ones)
                for param in resolved_operation_params:
                    if isinstance(param, dict) and 'name' in param:
                        param_name = param['name']
                        if param_name in seen_param_names:
                            # Remove existing parameter with same name
                            all_params = [p for p in all_params if p.get('name') != param_name]
                        all_params.append(param)
                        seen_param_names.add(param_name)

                # Add parameter descriptions to text
                for param in all_params:
                    if isinstance(param, dict):
                        param_name = param.get('name', '')
                        param_desc = param.get('description', '')
                        param_in = param.get('in', '')

                        param_text = f"parameter {param_name}"
                        if param_desc:
                            param_text += f": {param_desc}"
                        if param_in:
                            param_text += f" (in: {param_in})"
                        text_parts.append(param_text)

                # Add request body info if present
                request_body = operation_data.get('requestBody')
                if request_body and isinstance(request_body, dict):
                    content = request_body.get('content', {})
                    if isinstance(content, dict):
                        for media_type, media_type_data in content.items():
                            if isinstance(media_type_data, dict):
                                text_parts.append(f"request body {media_type}")

                                # Extract and resolve schema properties
                                schema = media_type_data.get('schema')
                                if schema and isinstance(schema, dict):
                                    # Resolve $ref in schema
                                    resolved_schema = self._resolve_schema(schema)
                                    schema_desc = resolved_schema.get('description', '')
                                    if schema_desc:
                                        text_parts.append(schema_desc)

                                    # Extract nested properties with dot notation from resolved schema
                                    body_properties = self._extract_schema_properties(resolved_schema)
                                    for prop_name in body_properties:
                                        text_parts.append(f"body parameter {prop_name}")

                # Combine all text
                combined_text = " ".join(text_parts)

                endpoints.append({
                    'text': combined_text,
                    'path': path_url,
                    'method': method.upper(),
                    'summary': summary,
                    'description': description,
                    'operation_id': operation_id,
                    'path_data': path_data,
                    'operation_data': operation_data,
                    'path_params': resolved_path_params,
                    'operation_params': resolved_operation_params,
                    'merged_params': all_params,
                    'request_body': request_body,
                })

        return endpoints

    def _calculate_cfidf_similarity(self, query: str) -> list[tuple[float, dict[str, any]]]:
        """
        Calculate CF-IDF similarity between query and endpoints.
        :param query: Query text
        :return: List of (similarity_score, endpoint) tuples
        """
        # Normalize and tokenize query
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        query_tokens = query.split()

        # Extract all endpoint texts
        endpoint_texts = [
            re.sub(r'[^a-zA-Z0-9\s]', '', endpoint['text'].lower()).split() for endpoint in self.endpoints]

        # Build vocabulary
        vocabulary = set()
        for tokens in endpoint_texts:
            vocabulary.update(tokens)
        vocabulary.update(query_tokens)
        vocabulary = list(vocabulary)

        # Calculate document frequency (DF) for each term
        df = {}
        for term in vocabulary:
            df[term] = sum(1 for tokens in endpoint_texts if term in tokens)

        # Calculate IDF for each term
        N = len(self.endpoints)  # Total number of endpoints
        idf = {term: math.log(N / (1 + df[term])) for term in vocabulary}

        # Calculate CF-IDF vectors for each endpoint
        endpoint_vectors = []
        for tokens in endpoint_texts:
            # Calculate class frequency (CF) for each term in this endpoint
            cf = Counter(tokens)

            # Calculate CF-IDF vector
            vector = np.zeros(len(vocabulary))
            for i, term in enumerate(vocabulary):
                if term in cf:
                    vector[i] = cf[term] * idf[term]

            endpoint_vectors.append(vector)

        # Calculate CF-IDF vector for query
        query_cf = Counter(query_tokens)
        query_vector = np.zeros(len(vocabulary))
        for i, term in enumerate(vocabulary):
            if term in query_cf:
                query_vector[i] = query_cf[term] * idf[term]

        # Calculate cosine similarity between query and each endpoint
        from sklearn.metrics.pairwise import cosine_similarity
        query_vector = query_vector.reshape(1, -1)  # Reshape for cosine_similarity function
        similarities = cosine_similarity(query_vector, endpoint_vectors)[0]

        # Pair similarities with endpoints
        scored_endpoints = [(float(similarities[i]), endpoint) for i, endpoint in enumerate(self.endpoints)]

        return scored_endpoints

    def _format_endpoint_dict(self, endpoint: dict[str, any]) -> dict[str, any]:
        """
        Format endpoint information into a structured dictionary.
        :param endpoint: Endpoint dictionary with metadata
        :return: Formatted dictionary with endpoint information
        """
        operation_data = endpoint['operation_data']

        # Create the base structure
        result = {
            "path": endpoint['path'],
            "method": endpoint['method'],
        }

        # Add summary if present
        if endpoint['summary']:
            result["summary"] = endpoint['summary']

        # Add description if present
        if endpoint['description']:
            result["description"] = endpoint['description']

        # Parameters (already deduplicated by name)
        parameters = []
        for param in endpoint['merged_params']:
            if isinstance(param, dict):
                param_dict = {
                    "name": param.get('name', ''),
                    "in": param.get('in', ''),
                    "required": param.get('required', False)
                }

                # Determine type from schema
                schema = param.get('schema', {})
                if isinstance(schema, dict):
                    param_type = schema.get('type', 'string')
                else:
                    param_type = 'string'
                param_dict["type"] = param_type

                # Add description if present
                description = param.get('description', '')
                if description:
                    param_dict["description"] = description

                parameters.append(param_dict)

        if parameters:
            result["parameters"] = parameters

        # Request Body
        request_body = endpoint['request_body']
        if request_body and isinstance(request_body, dict):
            content = request_body.get('content', {})
            if isinstance(content, dict):
                request_body_dict = {}
                for media_type, media_type_data in content.items():
                    if isinstance(media_type_data, dict):
                        schema = media_type_data.get('schema', {})
                        if isinstance(schema, dict):
                            # Resolve $ref in schema before formatting
                            resolved_schema = self._resolve_schema(schema)
                            schema_info = self._format_schema(resolved_schema)
                            # Add nested properties with dot notation from resolved schema
                            body_properties = self._extract_schema_properties(resolved_schema)
                            if body_properties:
                                schema_info["properties_flat"] = body_properties
                            request_body_dict[media_type] = schema_info

                if request_body_dict:
                    result["requestBody"] = {
                        "required": request_body.get('required', False),
                        "content": request_body_dict
                    }

        # Responses => don't include them, they are not really relevant
        # responses = {}
        # responses_data = operation_data.get('responses', {})
        # if isinstance(responses_data, dict):
        #     for status_code, response_data in responses_data.items():
        #         if isinstance(response_data, dict):
        #             response_desc = response_data.get('description', 'No description')
        #             responses[status_code] = response_desc
        #         else:
        #             responses[status_code] = str(response_data)
        # 
        # if responses:
        #     result["responses"] = responses

        return result

    def _format_schema(self, schema: dict[str, any]) -> dict[str, any]:
        """
        Format a schema object into a dictionary.
        :param schema: Schema dictionary
        :return: Schema dictionary
        """
        # First resolve any $ref in the schema itself
        resolved_schema = self._resolve_schema(schema)

        schema_dict = {
            "type": resolved_schema.get('type', 'string')
        }

        if 'description' in resolved_schema and resolved_schema['description']:
            schema_dict["description"] = resolved_schema['description']

        # Add properties for object schemas
        properties = resolved_schema.get('properties', {})
        if isinstance(properties, dict) and properties:
            properties_dict = {}
            for prop_name, prop_schema in properties.items():
                if isinstance(prop_schema, dict):
                    # Recursively format the property schema (this will also resolve $ref)
                    properties_dict[prop_name] = self._format_schema(prop_schema)
            schema_dict["properties"] = properties_dict

            if 'required' in resolved_schema and isinstance(resolved_schema['required'], list):
                schema_dict["required"] = resolved_schema['required']

        # Handle allOf schemas by merging properties
        all_of = resolved_schema.get('allOf', [])
        if isinstance(all_of, list) and all_of:
            # For allOf, we need to merge properties from all sub-schemas
            merged_properties = {}
            merged_required = []

            for sub_schema in all_of:
                if isinstance(sub_schema, dict):
                    # Recursively format the sub-schema
                    formatted_sub_schema = self._format_schema(sub_schema)

                    # Merge properties
                    sub_properties = formatted_sub_schema.get('properties', {})
                    if isinstance(sub_properties, dict):
                        merged_properties.update(sub_properties)

                    # Merge required fields
                    sub_required = formatted_sub_schema.get('required', [])
                    if isinstance(sub_required, list):
                        merged_required.extend(sub_required)

            if merged_properties:
                schema_dict["properties"] = merged_properties

            if merged_required:
                # Remove duplicates while preserving order
                seen = set()
                unique_required = []
                for field in merged_required:
                    if field not in seen:
                        seen.add(field)
                        unique_required.append(field)
                schema_dict["required"] = unique_required

        # Add items for array schemas
        items = resolved_schema.get('items', {})
        if isinstance(items, dict) and items:
            schema_dict["items"] = self._format_schema(items)

        # If this schema has properties, it should be an object type
        if 'properties' in schema_dict and schema_dict['properties']:
            schema_dict['type'] = 'object'

        return schema_dict

    def _extract_schema_properties(self, schema: dict[str, any], prefix: str = "", visited: set = None,
                                   max_depth: int = 1, current_depth: int = 0) -> list[str]:
        """
        Recursively extract property names from a schema with dot notation.
        :param schema: Schema dictionary to extract properties from
        :param prefix: Current property path prefix
        :param visited: Set of visited schemas to prevent infinite recursion
        :param max_depth: Maximum depth to extract properties (default: 2)
        :param current_depth: Current recursion depth
        :return: List of property names with dot notation
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        schema_id = id(schema)
        if schema_id in visited:
            return []
        visited.add(schema_id)

        # Limit depth to prevent excessive property extraction
        if current_depth >= max_depth:
            return []

        properties = []

        # Handle allOf schemas by merging properties from all sub-schemas
        all_of = schema.get('allOf', [])
        if isinstance(all_of, list):
            for sub_schema in all_of:
                if isinstance(sub_schema, dict):
                    nested_props = self._extract_schema_properties(
                        sub_schema, prefix, visited, max_depth, current_depth)
                    properties.extend(nested_props)

        # Check if schema has properties
        schema_properties = schema.get('properties', {})
        if isinstance(schema_properties, dict):
            # Get required fields
            required_fields = schema.get('required', [])
            if not isinstance(required_fields, list):
                required_fields = []

            for prop_name, prop_schema in schema_properties.items():
                # Add required marker if field is required
                current_path = f"{prefix}.{prop_name}" if prefix else prop_name
                if prop_name in required_fields:
                    current_path = f"{current_path} (required)"
                properties.append(current_path)

                # Recursively extract nested properties (only if we haven't reached max depth)
                if isinstance(prop_schema, dict) and current_depth < max_depth:
                    nested_props = self._extract_schema_properties(
                        prop_schema, current_path, visited, max_depth, current_depth + 1)
                    properties.extend(nested_props)

        # Handle array of objects (only if we haven't reached max depth)
        items = schema.get('items', {})
        if isinstance(items, dict) and items != schema and current_depth < max_depth:  # Prevent self-reference
            nested_props = self._extract_schema_properties(items, prefix, visited, max_depth, current_depth + 1)
            properties.extend(nested_props)

        return properties
