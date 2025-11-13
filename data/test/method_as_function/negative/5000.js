// This tests complex objects as array elements, only the first element is tested, attribute type is wrong
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/customers', {
    name: 'name',
    addresses: [
        {
            city: "Darmstadt",
            house: {
                number: "2",
                street: "street"
            }
        }
    ]
});