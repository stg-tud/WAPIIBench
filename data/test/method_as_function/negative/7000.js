// This tests if string concatenation is allowed (it shouldn't)
const axios = require('axios');
const name = "johndoe"
axios.get('https://petstore.swagger.io/v2/customers/' + name);