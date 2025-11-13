// This tests schemas without type
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/customers', {
    bumpscosity: 76,
    name: 'name',
    addresses: [
        {
            city: "Darmstadt",
            house: {
                number: 1,
                street: "street"
            }
        }
    ]
});