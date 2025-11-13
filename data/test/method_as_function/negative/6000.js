// This tests API key authentication, with no API provided
const axios = require('axios');
const id = 42
axios.delete(`https://petstore.swagger.io/v2/pets-secure/${id}`);