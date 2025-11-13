Consider this OpenAPI specification:

```yaml
{spec}
```

I want you to generate test data for this API. The data should be in JSON format and look like this:

```json
{json_template}
```

* `samples` is an array containing all the test cases.
* `task` is a natural language description of a specific task that can be solved by sending a request to the given API. All information required to unambiguously identify and implement the corresponding API request must be contained in the task description. Therefore, state all expected argument values explicitly. The only exception is authentication keys and tokens, which must not be specified here (but you may specify the authentication method to be used if multiple ones are available).
* `config` is an object containing the expected configuration of the described request. If a property of `config` is empty, it can be omitted.
* `url` is the full URL of the endpoint (server URL + path) and may include path parameters but no query parameters.
* `method` is the HTTP method used. It can be one of `get`, `put`, `post`, `delete`, and `patch`.
* `headers` is an object containing all expected header arguments. Note that the property `"Accept": "application/json, text/plain, */*"` is always present and if `data` is sent in the request body, `"Content-Type": "mime/type"` is present as well (substitute "mime/type" for the respective media type).
* `params` is an object containing all expected query arguments.
* `data` is an object containing all data from the request body. It is only used for the methods put, post, delete, and patch.

Remember that for authentication, an additional argument might be required. Take a look at an endpoint's `security` property and the `securitySchemes` section in the specification, to find out if and how to authenticate. I assume you know how the usual authentication schemes `apiKey`, `http`, and `oauth2` work. Use `<key>` as a placeholder for API keys (e.g., `"name": "<key>"`) and `<token>` as a placeholder for authorization tokens (e.g., `"Authorization": "Bearer <token>"`).
{api_specific_instructions}
Now, please give me a JSON object that matches my description and that contains diverse examples of requests to this API. {path_selection}
