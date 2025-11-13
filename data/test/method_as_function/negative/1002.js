const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/pets', {
    foo: 'name',
    bar: 'tag'
});