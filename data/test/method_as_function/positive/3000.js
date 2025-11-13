const axios = require('axios');
const id = 42
axios.delete(`https://petstore.swagger.io/v2/pets/${id}`);