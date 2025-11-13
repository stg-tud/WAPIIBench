// This tests a request with missing required query param
const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/customers');