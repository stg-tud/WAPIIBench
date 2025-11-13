const axios = require('axios');
const id = 42
axios.get(`https://petstore_swagger.io/v2/pets/${id}`);