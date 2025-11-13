// This tests empty data/config when no parameters are defined
const axios = require('axios');
const id = 42
axios.get(`https://petstore.swagger.io/v2/pets/${id}`, {});