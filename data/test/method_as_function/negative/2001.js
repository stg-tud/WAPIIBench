const axios = require('axios');
const id = 42
axios.get(`https://petstore.swagger.io/v2/pets/${id}`, {
   params: {
       tags: ['tag1', 'tag2'],
       limit: 10
   }
});