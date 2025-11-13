from __future__ import annotations

import logging
import os
import sys
import uuid
from collections import defaultdict
from enum import auto, Enum
from typing import NamedTuple

import regex as re
from openapi_parser import parse
from openapi_parser.enumeration import BaseLocation, DataType, OperationMethod, ParameterLocation, SecurityType
from openapi_parser.specification import AnyOf, Array, Boolean, Integer, Number, Object, OneOf, Operation, Parameter, \
    Path, Property, RequestBody, Schema, Security, Specification, String

from generation_rules import GenerationRule, GenerationRuleset

logging.basicConfig(format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class Request(NamedTuple):
    path: str
    method: OperationMethod
    parameters: list[Parameter]
    auth: list[Security] | None
    request_body: RequestBody | None


class AxiosSyntax(Enum):
    METHOD_AS_FUNCTION = auto()
    ALL_IN_BODY_URL_FIRST = auto()


# Define some regex building blocks
INTEGER_VAL = r"-?\d+"
NUMBER_VAL = r"-?\d+(?:\.\d+)?"
STRING_VAL = r"""(?:"(?:[^"\\]|(?:\\.))*"|'(?:[^'\\]|(?:\\.))*'|`(?:[^`\\]|(?:\\.))*`)"""  # string interpolation is not handled explicitly
STRING_VAL_IN_URL = r"[a-zA-Z0-9_.~<>:-]+"  # <, >, and : are not allowed in typical URLs but needed in our dataset
BOOLEAN_VAL = r"(?:true|false)"
VAR_NAME = r"[a-zA-Z_$][a-zA-Z0-9_$.()]*"  # includes dereferencing and method invocation
TEMPLATE_STRING_VAR = fr"\$\{{{VAR_NAME}\}}"

MAX_OBJECTS_IN_ARRAY = 5
MAX_RECURSION_DEPTH = 3

_spec_cache = {}


def parse_spec(file_path: str, strict_enum: bool = False) -> Specification:
    """
    Wrapper for openapi_parser.parse which implements caching.
    :param file_path: Path or URL to OpenAPI file
    :param strict_enum: Validate content types and string formats against the enums defined in openapi-parser.
        Note that the OpenAPI specification allows for custom values in these properties.
    :return: The specification object
    """
    spec = _spec_cache.get(file_path)
    if spec is not None:
        logger.info(f"Using cached spec for {file_path}")
        return spec

    logger.info(f"Cache miss - parsing spec for {file_path}")
    spec = parse(uri=file_path, strict_enum=strict_enum)
    _spec_cache[file_path] = spec
    return spec


def spec_to_ruleset(file_path: str, axios_syntax: AxiosSyntax = AxiosSyntax.METHOD_AS_FUNCTION) -> GenerationRuleset:
    """
    Convert an OpenAPI specification to a generation ruleset that describes valid requests to that API.
    This method is tailored specifically to the JavaScript axios library.
    :param file_path: Path to the specification file
    :param axios_syntax: The syntax to use for Axios calls
    :return: A list of generation rules describing valid requests
    """
    logger.info(f"Converting OpenAPI spec to generation ruleset: {file_path=}, {axios_syntax=}")

    servers, possible_requests = _get_possible_requests(file_path)

    servers = [re.escape(server) for server in servers]
    server_regex = _join_alternatives(servers, inner_parentheses=False, outer_parentheses=False)

    if axios_syntax is AxiosSyntax.METHOD_AS_FUNCTION:
        ruleset = _create_method_as_function_ruleset(possible_requests, server_regex)
    elif axios_syntax is axios_syntax.ALL_IN_BODY_URL_FIRST:
        ruleset = _create_all_in_body_url_first_ruleset(possible_requests, server_regex)
    else:
        raise ValueError(f"Unsupported Axios syntax {axios_syntax}")

    logger.info("Ruleset created successfully")
    return ruleset


def _create_all_in_body_url_first_ruleset(possible_requests: dict[str, list[Request]],
                                          server_regex: str) -> GenerationRuleset:
    """
    Create a ruleset that follows the call syntax ``axios.request(config)`` with URL as first argument in ``config``.
    :param possible_requests: List of possible requests for the API
    :param server_regex: Regex of the URL's server part
    :return: The ruleset
    """
    possible_requests_by_path = defaultdict(list)
    for _, requests in possible_requests.items():
        for request in requests:
            possible_requests_by_path[request.path].append(request)

    generation_ruleset = GenerationRuleset()

    # Convert the possible requests into regexes and the regexes into generation rules
    for path, requests in possible_requests_by_path.items():
        logger.debug(f"Converting {len(requests)} requests for path [{path}]")
        request_regexes = []
        url_regex = server_regex + _create_path_regex(requests[0])

        for request in requests:
            logger.debug(f"Converting request: {request.path} [{request.method.name}]")

            arguments_regex = _create_arguments_regex(request, AxiosSyntax.ALL_IN_BODY_URL_FIRST)

            statement_regex = fr"method:\s*'{request.method.value}'{arguments_regex}"

            request_regexes.append(statement_regex)

        request_regex = _join_alternatives(request_regexes)

        starter = fr"""axios\.request\(\{{\s*url:\s*(?:"|'|`){url_regex}(?:"|'|`)"""  # URL is currently not constrained
        stopper = r"}\)\s*(?:;|\.then\(|\.catch\(|\.finally\()"  # asserting that parentheses are balanced is too complicated
        body = fr",\s*{request_regex}\s*{stopper}"

        generation_ruleset.append(GenerationRule(starter, stopper, body, name=path))

    return generation_ruleset


def _create_method_as_function_ruleset(possible_requests: dict[str, list[Request]],
                                       server_regex: str) -> GenerationRuleset:
    """
    Create a ruleset that follows the call syntax ``axios.method(url[, config])``.
    :param possible_requests: List of possible requests for the API
    :param server_regex: Regex of the URL's server part
    :return: The ruleset
    """
    generation_ruleset = GenerationRuleset()
    # The funnel rule guides the generation to trigger any of the actual rules
    generation_ruleset.append(GenerationRule(
        r"axios\.", r"\(", _join_alternatives(list(possible_requests.keys()), inner_parentheses=False) + r"\(",
        name="funnel"))

    # Convert the possible requests into regexes and the regexes into generation rules
    for method, requests in possible_requests.items():
        logger.debug(f"Converting {len(requests)} requests for method [{method}]")

        request_regexes = []

        for request in requests:
            logger.debug(f"Converting request: {request.path} [{request.method.name}]")
            url_regex = server_regex + _create_path_regex(request)

            arguments_regex = _create_arguments_regex(request)

            statement_regex = fr"""(?:"|'|`){url_regex}(?:"|'|`){arguments_regex}"""  # we rely on the model to choose matching quotes

            request_regexes.append(statement_regex)

        request_regex = _join_alternatives(request_regexes)

        starter = fr"axios\.{method}\("
        stopper = r"\)\s*(?:;|\.then\(|\.catch\(|\.finally\()"  # asserting that parentheses are balanced is too complicated
        body = fr"\s*{request_regex}\s*{stopper}"
        generation_ruleset.append(GenerationRule(starter, stopper, body, name=method.upper()))

    return generation_ruleset


def _get_possible_requests(file_path: str) -> tuple[list[str], dict[str, list[Request]]]:
    """
    Read the specification, extract all relevant information, and pack it into suitable data structures.
    :param file_path: Path to the specification file
    :return: A list of server urls and a dictionary of possible requests for each method
    """
    logger.info(f"Parsing OpenAPI spec: {file_path=}")

    spec = parse_spec(file_path)

    server_urls = [server.url for server in spec.servers]
    global_securities = {name for security in spec.security for name in security.keys()}

    # Axios supports explicitly setting a Content-Type header to cause automatic serialization of request body to a specific media type
    content_type_header = Parameter(
        "Content-Type", ParameterLocation.HEADER, schema=String(DataType.STRING), required=False)

    # Extract all possible requests and group them by HTTP method
    possible_requests = defaultdict(list)
    for path in spec.paths:
        for operation in path.operations:

            method = operation.method
            logger.debug(f"Adding request: [{method.name}] {path.url} ({operation.operation_id})")

            operation_securities = {name for security in operation.security for name in security.keys()}
            if global_securities or operation_securities:
                logger.debug("Operation requires authentication")

            auth = []
            for security in global_securities | operation_securities:
                logger.debug(f" - {security}")

                security_schema = spec.security_schemas[security]

                logger.debug(f"   - Type: {security_schema.type.name}")
                logger.debug(f"   - Scheme: {security_schema.scheme.name if security_schema.scheme else None}")

                if not security_schema.location:
                    security_schema.location = BaseLocation.HEADER
                logger.debug(f"   - Location: {security_schema.location.name}")

                if not security_schema.name:
                    security_schema.name = "Authorization"
                logger.debug(f"   - Name: {security_schema.name}")

                auth.append(security_schema)

            parameters = path.parameters + operation.parameters
            if method in [OperationMethod.PUT, OperationMethod.POST, OperationMethod.PATCH]:
                logger.debug("Adding Content-Type header")
                parameters.append(content_type_header)

            request = Request(path.url, method, parameters, auth, operation.request_body)
            possible_requests[method.value].append(request)

    return server_urls, possible_requests


def _create_path_regex(request: Request) -> str:
    """
    Create a regex that describes the path in the URL for the given request, including path and (potentially) query parameters.
    :param request: The request for which the URL is being constructed
    :return: A regex describing the URL path for the given request
    """
    logger.debug(f"Creating path regex for request: {request.path} [{request.method.name}]")

    path_regex = re.escape(request.path)
    query_params = []

    for param in request.parameters:

        location = param.location
        if location is ParameterLocation.PATH:
            logger.debug(f"Processing path parameter: {param.name} | {param.location.name}")
            # Replace the placeholder in the path with a regex for either a value or string interpolation
            url_param_regex = _create_val_or_var_regex(param.schema, True)
            # Passing a lambda as repl argument is a workaround to prevent the backslash escapes in it from being processed
            path_regex = re.sub(fr"\\\{{{re.escape(param.name)}\\\}}", lambda matchobj: url_param_regex, path_regex)
        elif location is ParameterLocation.QUERY:
            # Collect all query parameters first and process them later
            query_params.append(param)

    return path_regex

    # The following lines add the URL's query parameter suffix.
    # We don't use it for now as it is better practice to pass query parameters through the `params` argument.
    # noinspection PyUnreachableCode
    query_key_value_list_regex, _ = _create_key_value_list_regex(query_params, r"\?", "&", "", "=", "query")
    return path_regex + query_key_value_list_regex


def _create_arguments_regex(request: Request, axios_syntax: AxiosSyntax = AxiosSyntax.METHOD_AS_FUNCTION) -> str:
    """
    Create a regex that describes the arguments passed to the axios call, including request body and query/header parameters.
    :param request: The request for which to build the regex
    :return: A regex describing the call arguments, or an empty string if there is neither a body nor a config
    """
    logger.debug(f"Creating arguments regex for request: {request.path} [{request.method.name}]")

    # Build a regex for the request body
    body = request.request_body
    if body:  # Only certain methods do have a request body
        logger.debug(f"generating regex for request body")
        assert len(body.content) == 1, f"We assume there is always just one body, but we got {len(body.content)}"
        content = body.content[0]

        if not body.required:
            logger.warning(f"Optional request body not supported: {request.method=}, {request.path=}")

        content_schema = content.schema
        assert isinstance(content_schema, Object), f"Unsupported request body {content_schema=}"

        body_key_value_list_regex, _ = _create_key_value_list_regex(
            content_schema.properties, "", r",\s*", r",?", r":\s*", "body", required_params=content_schema.required)
        data_regex = fr"(?:\{{\s*{body_key_value_list_regex}\s*\}}|{VAR_NAME})"
    elif request.method in [OperationMethod.PUT, OperationMethod.POST, OperationMethod.PATCH]:
        # These methods require a request body, even if none is defined in the spec
        data_regex = r"(?:null|\{\})"
    else:
        data_regex = None

    # Build a regex for the header params
    header_params = [param for param in request.parameters if param.location is ParameterLocation.HEADER]

    for security in request.auth:
        if security.location is BaseLocation.HEADER:
            required = len(request.auth) == 1  # Multiple security schemes are not supported properly yet

            logger.debug(f"Processing security parameter: {security.name} | {security.location.name}")
            header_params.append(
                Parameter(security.name, ParameterLocation.HEADER, schema=String(DataType.STRING), required=required))

    if header_params:
        header_key_value_list_regex, has_required_header_params = _create_key_value_list_regex(
            header_params, "", r",\s*", r",?", r":\s*", "headers")
        header_regex = fr"headers:\s*\{{\s*{header_key_value_list_regex}\s*\}}"
        if not has_required_header_params:
            header_regex = fr"(?:{header_regex})?"
    else:
        header_regex = None
        has_required_header_params = False

    # Build a regex for the query params
    query_params = [param for param in request.parameters if param.location is ParameterLocation.QUERY]

    for security in request.auth:
        if security.location is BaseLocation.QUERY:
            required = len(request.auth) == 1  # Multiple security schemes are not supported properly yet

            logger.debug(f"Processing security parameter: {security.name} | {security.location.name}")
            query_params.append(
                Parameter(security.name, ParameterLocation.QUERY, schema=String(DataType.STRING), required=required))

    if query_params:
        query_key_value_list_regex, has_required_query_params = _create_key_value_list_regex(
            query_params, "", r",\s*", r",?", r":\s*", "params")
        query_regex = fr"params:\s*\{{\s*{query_key_value_list_regex}\s*\}}"
        if not has_required_query_params:
            query_regex = fr"(?:{query_regex})?"
    else:
        query_regex = None
        has_required_query_params = False

    # Combine header and query params to the complete config object
    if header_regex and query_regex:
        # The ",?" permits omitting the comma even if it's required, but we rely on the model to still get it right
        config_regex = fr"{header_regex},?\s*{query_regex}"
        has_required_config_params = has_required_header_params or has_required_query_params
    elif header_regex:
        config_regex = fr"{header_regex}"
        has_required_config_params = has_required_header_params
    elif query_regex:
        config_regex = fr"{query_regex}"
        has_required_config_params = has_required_query_params
    else:
        config_regex = None
        has_required_config_params = False

    if axios_syntax is AxiosSyntax.METHOD_AS_FUNCTION and config_regex:
        config_regex = fr"\{{\s*{config_regex}\s*\}}"

    if axios_syntax is AxiosSyntax.ALL_IN_BODY_URL_FIRST and data_regex:
        data_regex = fr"data:\s*{data_regex}"

    # Combine data and config to the complete argument list
    if data_regex and config_regex:
        if not has_required_config_params:
            arguments_regex = fr",\s*{data_regex}(?:,\s*{config_regex})?"
        else:
            arguments_regex = fr",\s*{data_regex},\s*{config_regex}"
    elif data_regex:
        arguments_regex = fr",\s*{data_regex}"
    elif config_regex:
        if not has_required_config_params:
            arguments_regex = fr"(?:,\s*{config_regex})?"
        else:
            arguments_regex = fr",\s*{config_regex}"
    else:
        arguments_regex = ""

    return arguments_regex


def _create_key_value_list_regex(params: list[Parameter] | list[Property], starter: str, separator: str,
                                 terminator: str, assignment: str, uid: str, required_params: list[str] | None = None,
                                 recursion_depth: int = 0) -> tuple[str, bool]:
    """
    Helper function to create a regex that matches a sequence of assignments from values to keys. The sequence may
    contain these key-value pairs in any order, but each pair at most once and required pairs exactly once.
    :param params: The items to be included in the sequence. The type of the values is extracted from their schemas
    :param starter: A prefix for the sequence (may be empty)
    :param separator: A separator between the key-value pairs
    :param terminator: A suffix for the sequence (may be empty)
    :param assignment: A separator between each key and value
    :param uid: A unique name for this sequence used to name a capturing group
    :param required_params: List of required parameters. Only needed if ``params`` contains Property objects
    :param recursion_depth: Current recursion depth
    :return: A regex matching a sequence of optional and required key-value assignments
    """
    if not params:
        return "", False

    guid = f"{uid}_{uuid.uuid4().hex[:8]}"
    separator_regex = fr"(?(_{guid}){separator}|(?P<_{guid}>{starter}))"

    all_param_regexes = []

    for param in params:
        logger.debug(f"Processing parameter {param.name} | {_get_param_location(param)}")

        param_guid = f"{guid}_{_sanitize_name(param.name)}"
        param_name_regex = re.escape(param.name)
        param_name_regex = fr"""(?:{param_name_regex}|"{param_name_regex}"|'{param_name_regex}')"""
        param_regex = fr"""{param_name_regex}{assignment}{_create_val_or_var_regex(param.schema, False, recursion_depth)}"""

        # The following regex reads as:
        # If the capture group belonging to the parameter is not defined, then define it and match the parameter
        # If the capture group is already defined, then there is no match. This ensures that a parameter is only matched once
        all_param_regexes.append(fr"(?({param_guid})|(?P<{param_guid}>{param_regex}))")

    all_param_regex = fr"(?:{separator_regex}{_join_alternatives(all_param_regexes, inner_parentheses=False)})*"

    if required_params is not None:
        is_required = lambda param: param.name in required_params
    else:
        is_required = lambda param: param.required is not None and param.required

    required_param_regexes = []

    for param in params:
        if not is_required(param):
            continue
        logger.debug(f"Processing required parameter {param.name} | {_get_param_location(param)}")

        # (?!) is an impossible match. So the following regex can be read as:
        # If the capture group belonging to the parameter is not defined, then try to match the impossible
        required_param_regexes.append(fr"(?({guid}_{_sanitize_name(param.name)})|(?!))")

    required_param_regex = "".join(required_param_regexes)

    has_required_params = len(required_param_regexes) > 0

    return f"{all_param_regex}{required_param_regex}{terminator}", has_required_params


def _sanitize_name(name: str) -> str:
    """
    Make sure no bad characters appear in a group name.
    :param name: The name to sanitize
    :return: A sanitized version of the name
    """
    return re.sub(r"\W", "_", name)


def _get_param_location(param: Parameter) -> str:
    """
    Helper function to get the location of a parameter as string.
    :param param: The Parameter object
    :return: The location as string
    """
    return param.location.name if hasattr(param, 'location') else "BODY"


def _create_val_or_var_regex(schema: Schema, in_url: bool, recursion_depth: int = 0) -> str:
    """
    Helper function to create a regex that matches either a literal value or a variable.
    :param schema: The schema of the value (determines the data type)
    :param in_url: If true, variables are placed through string interpolation
    :param recursion_depth: Current recursion depth
    :return: A regex matching either a value or a variable
    """
    # Actually, string interpolation is only allowed within backtick quotes, but we rely on the model to get this right.
    return fr"(?:{_schema_to_regex(schema, in_url=in_url, recursion_depth=recursion_depth)}|{TEMPLATE_STRING_VAR if in_url else VAR_NAME})"


def _schema_to_regex(schema: Schema, in_url: bool = False, recursion_depth: int = 0) -> str:
    """
    Create a regex that describes the instances of the datatype given by schema.
    :param schema: Schema of the datatype
    :param in_url: If true, regex describes a value inside a URL string
    :param recursion_depth: Current recursion depth
    :return: A regex describing instances of the datatype
    """
    logger.debug(f"Creating regex for schema type {schema.type.name}, {in_url=}")

    if isinstance(schema, Integer):
        return INTEGER_VAL
    elif isinstance(schema, Number):
        return NUMBER_VAL
    elif isinstance(schema, String):
        return STRING_VAL_IN_URL if in_url else STRING_VAL
    elif isinstance(schema, Boolean):
        return BOOLEAN_VAL
    elif isinstance(schema, Array):
        if in_url:
            raise ValueError("Array parameters in URLs are not supported")
        return _create_array_regex(schema, recursion_depth)
    elif isinstance(schema, Object):
        if in_url:
            raise ValueError("Object parameters in URLs are not supported")
        return _create_object_regex(schema, recursion_depth)
    elif isinstance(schema, OneOf) or isinstance(schema, AnyOf):
        schema_regexes = [_schema_to_regex(s, in_url=in_url, recursion_depth=recursion_depth) for s in schema.schemas]
        return _join_alternatives(schema_regexes, inner_parentheses=False)
    else:
        raise ValueError(f"Unsupported {schema=}")


def _create_array_regex(schema: Array, recursion_depth: int = 0) -> str:
    """
    Create a regex that describes the instances of an array datatype.
    :param schema: Schema of the array datatype
    :param recursion_depth: Current recursion depth
    """
    if schema.items is None:
        logger.warning(f"Array schema without items: {schema}")
        return r"\[\]"

    logger.debug(f"Creating regex for array schema with item type {schema.items.type.name}")

    if recursion_depth >= MAX_RECURSION_DEPTH:
        logger.warning("Maximum recursion depth exceeded, returning wildcard regex")
        return r"\[[^[\]]*\]"  # Allow everything except another nested array

    # To make objects with required and not-required attributes work, regex groups need to be instantiated with unique names.
    # This is a workaround to make that happen. Look at the _create_key_value_list_regex function to see how this is used.
    content_regex = _create_val_or_var_regex(schema.items, False, recursion_depth + 1)
    for i in range(MAX_OBJECTS_IN_ARRAY - 1):
        content_regex = fr"{content_regex}(?:,\s*{_create_val_or_var_regex(schema.items, False, recursion_depth + 1)})?"

    # Array content is optional to handle empty arrays
    return fr"\[\s*(?:{content_regex})?\s*\]"


def _create_object_regex(schema: Object, recursion_depth: int = 0) -> str:
    """
    Create a regex that describes the instances of an object datatype.
    :param schema: Schema of the object datatype
    :param recursion_depth: Current recursion depth
    """
    if not schema.properties:
        logger.warning(f"Object schema without properties: {schema}")
        return r"\{\}"

    logger.debug(f"Creating regex for object schema with {len(schema.properties)} properties")

    if recursion_depth >= MAX_RECURSION_DEPTH:
        logger.warning("Maximum recursion depth exceeded, returning wildcard regex")
        return r"\{[^{}]*\}"  # Allow everything except another nested object

    key_value_list_regex, _ = _create_key_value_list_regex(
        schema.properties, "", r",\s*", r",?", r":\s*", "object", schema.required, recursion_depth + 1)

    return fr"\{{\s*{key_value_list_regex}\s*\}}"


def _join_alternatives(alternatives: list[str], inner_parentheses: bool = True, outer_parentheses: bool = True) -> str:
    """
    Join multiple regexes with an OR condition, making sure that parentheses are placed correctly.
    :param alternatives: The alternative regexes
    :param inner_parentheses: Whether to add parentheses around the individual alternatives
    :param outer_parentheses: Whether to add parentheses around the list of alternatives.
        Ignored unless the list contains exactly one element, as it would be too error-prone
    :return: A single regex matching any of the given alternatives
    """
    if len(alternatives) == 0:
        return ""
    if len(alternatives) == 1:
        return f"({alternatives[0]})" if outer_parentheses else alternatives[0]
    elif inner_parentheses:
        return f"(({')|('.join(alternatives)}))"
    else:
        return f"({'|'.join(alternatives)})"


def validate_argument(arg_name: str, field_name: str, method: str, path: Path | None,
                      security_schemas: dict[str, Security]) -> bool:
    """
    Check if the given argument is a valid parameter for the given endpoint.
    :param arg_name: The name of the argument to check
    :param field_name: The field name in the Axios config ('headers', 'params', 'path_params', or 'data')
    :param method: The used HTTP method
    :param path: The Path object of the endpoint
    :param security_schemas: Allowed authentication schemes
    :return: Whether this argument is a valid parameter
    """
    # Certain header/security params are not explicitly listed in the spec but are still valid under certain conditions
    if field_name == 'headers' and arg_name == "Accept":
        return True
    if field_name == 'headers' and arg_name == "Content-Type":
        return method in ["put", "post", "patch"]
    for security in security_schemas.values():
        if security.type is SecurityType.API_KEY:
            if field_name == location_to_field_name(security.location) and arg_name == security.name:
                return True
        elif security.type is SecurityType.HTTP or security.type is SecurityType.OAUTH2:
            if field_name == 'headers' and arg_name == "Authorization":
                return True
        elif security.type is SecurityType.OPEN_ID_CONNECT:
            logger.error("OpenID Connect Discovery is not supported - ignoring it")

    if path is None:
        return False

    if field_name == 'data':
        # Is the argument in the properties of this operation?
        for operation in path.operations:
            if operation.method.value == method:
                if operation.request_body is None or not operation.request_body.content:
                    break
                assert len(operation.request_body.content) == 1
                assert isinstance(operation.request_body.content[0].schema, Object)
                # noinspection PyUnresolvedReferences
                properties = operation.request_body.content[0].schema.properties
                for property in properties:
                    if property.name == arg_name:
                        return True
                break

    else:
        def check_parameters(params: list[Parameter]) -> bool:
            """
            Check if the argument we are looking for is in this parameter list.
            :param params: List of parameters
            :return: If the argument in this function's closure is in `params`
            """
            for param in params:
                if param.name == arg_name:
                    if location_to_field_name(param.location) == field_name:
                        return True
                    break
            return False

        # Is the argument in the parameters of this path?
        if check_parameters(path.parameters):
            return True

        # Is the argument in the parameters of this operation?
        for operation in path.operations:
            if operation.method.value == method:
                if check_parameters(operation.parameters):
                    return True
                break

    return False


def find_path_in_spec(url: str, spec: Specification) -> list[Path]:
    """
    Try to find the path in the specification that corresponds to the given URL and return all matching candidates.
    If multiple paths match, order them so the one with the fewest path parameters and the longest URL comes first.
    This method assumes that there are no query parameters in the URL.
    :param url: The URL of the endpoint used for the request
    :param spec: The complete OpenAPI specification
    :return: Potentially empty list of Path objects that correspond to ``url``
    """
    assert len(spec.servers) == 1
    url_without_server = url.removeprefix(spec.servers[0].url)

    paths = []
    for path in spec.paths:
        path_regex, num_path_params = re.subn(r"{.*?}", r"[^/?&]+", path.url)
        if re.fullmatch(path_regex, url_without_server):
            paths.append((num_path_params, path))

    if not paths:
        return []

    # The secondary sorting key is mainly for the Google Sheets API where we need to make sure that the :<operation>
    # suffix is interpreted as part of the path and not as a path param.
    paths.sort(key=lambda elem: (elem[0], -len(elem[1].url)))
    return list(tuple(zip(*paths))[1])  # remove num_path_params and return the paths as list


def find_operation_in_path(method: str, path: Path) -> Operation | None:
    """
    Find the operation in the given path for the given method.
    :param method: The method
    :param path: The Path object
    :return: The Operation object that corresponds to ``method`` or ``None``
    """
    operations = [operation for operation in path.operations if operation.method.value == method]
    assert len(operations) <= 1, f"There should only be one operation for method {method} in path {path.url}"
    return operations[0] if operations else None


def location_to_field_name(location: ParameterLocation | BaseLocation) -> str:
    """
    Utility function to convert a ParameterLocation to the corresponding name/key in the Axios config.
    :param location: The location object
    :return: The corresponding field name
    """
    if location is ParameterLocation.HEADER or location is BaseLocation.HEADER:
        return "headers"
    if location is ParameterLocation.QUERY or location is BaseLocation.QUERY:
        return "params"
    if location is ParameterLocation.PATH:
        return "path_params"
    raise AssertionError(f"Unexpected location {location}")


def field_name_to_location_str(field_name: str) -> str:
    """
    Utility function to convert a name/key in the Axios config to the *string value* of the corresponding
    ParameterLocation. Intended mainly for printing messages and thus also treats 'data' like a location.
    :param field_name: The field name
    :return: The string value of the corresponding location object
    """
    if field_name == "headers":
        return ParameterLocation.HEADER.value
    if field_name == "params":
        return ParameterLocation.QUERY.value
    if field_name == "path_params":
        return ParameterLocation.PATH.value
    if field_name == "data":
        return "request body"
    raise AssertionError(f"Unexpected field name {field_name}")


def _test_regexes() -> None:
    spec_file = "openapi/test/petstore-expanded-modified.yaml"

    print(f"Testing {AxiosSyntax.METHOD_AS_FUNCTION.name} syntax")
    test_ruleset = spec_to_ruleset(spec_file, axios_syntax=AxiosSyntax.METHOD_AS_FUNCTION)
    test_case_dir = "data/test/method_as_function"
    _run_regex_tests(test_case_dir, test_ruleset)

    print(f"Testing {AxiosSyntax.ALL_IN_BODY_URL_FIRST.name} syntax")
    test_ruleset = spec_to_ruleset(spec_file, axios_syntax=AxiosSyntax.ALL_IN_BODY_URL_FIRST)
    test_case_dir = "data/test/all_in_body_url_first"
    _run_regex_tests(test_case_dir, test_ruleset)


def _run_regex_tests(test_case_dir: str, test_ruleset: GenerationRuleset):
    positive_test_case_dir = f"{test_case_dir}/positive/"
    negative_test_case_dir = f"{test_case_dir}/negative/"

    passes = 0
    fails = 0
    print("Running test cases ...\n")

    for file_name in os.listdir(positive_test_case_dir):
        file_path = os.path.join(positive_test_case_dir, file_name)
        if not os.path.isfile(file_path) or os.path.splitext(file_name)[1] != ".js":
            print(f"Skipping: {file_name}")
            continue
        with open(file_path, 'r') as file:
            test_case = file.read()
            match = test_ruleset.match_whole_code(test_case, excluded=["funnel"])
            if match:
                passes += 1
            else:
                fails += 1
                first_line = test_case.split("\n")[0].removeprefix("// ")
                print(f"False negative: {file_name} | {first_line}")

    for file_name in os.listdir(negative_test_case_dir):
        file_path = os.path.join(negative_test_case_dir, file_name)
        if not os.path.isfile(file_path) or os.path.splitext(file_name)[1] != ".js":
            print(f"Skipping: {file_name}")
            continue
        with open(file_path, 'r') as file:
            test_case = file.read()
            match = test_ruleset.match_whole_code(test_case, excluded=["funnel"])
            if match:
                fails += 1
                first_line = test_case.split("\n")[0].removeprefix("// ")
                print(f"False positive: {file_name} | {first_line}")
            else:
                passes += 1
    print(f"Total test cases:\t{passes + fails}\nPasses:\t\t{passes}\nFails:\t\t{fails}\n")


def _test_specs() -> None:
    print("Testing specs ...\n")

    passing = []
    failing = []

    spec_dir = "openapi/real_world_specs/"
    for spec_name in os.listdir(spec_dir):
        spec_path = os.path.join(spec_dir, spec_name)
        if not os.path.isfile(spec_path):
            continue

        try:
            spec_to_ruleset(spec_path)
            print(f"Successfully converted {spec_name} to ruleset.\n")
            passing.append(spec_name)
        except Exception as e:
            print(f"Failed to convert {spec_name} to ruleset:\n{e}\n")
            failing.append(spec_name)

    print(f"Passing specs: {passing}")
    print(f"Failing specs: {failing}")


if __name__ == '__main__':
    os.chdir(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))
    sys.path.append(os.getcwd())

    _test_regexes()
    print()
    _test_specs()
