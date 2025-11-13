// This tests illegal string path parameters
const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/customers/john'+'doe');