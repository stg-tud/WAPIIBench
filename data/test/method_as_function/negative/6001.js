// This tests Bearer authentication, with no API Key provided
const axios = require('axios');
const id = 42
axios.get(`https://petstore.swagger.io/v2/pets-secure/${id}`);