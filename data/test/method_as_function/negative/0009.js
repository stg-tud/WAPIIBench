// This tests if query parameters in the URL are allowed (they shouldn't)
const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/pets?limit=10');