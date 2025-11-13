const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/pets', {
    name: 'name',
    tag: 'tag'
});