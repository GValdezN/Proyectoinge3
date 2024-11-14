$(document).ready(function () {
    $('#imageUpload').change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').hide();
        $('.loader').hide();

        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
            $('#imagePreview').hide();
            $('#imagePreview').fadeIn(650);
        };
        reader.readAsDataURL(this.files[0]);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);

                // Actualizar resultados de Model 1
                $('#model1-class').text("Clase predicha: " + data.model_1.predicted_class);
                $('#model1-confidence').text("Confianza: " + data.model_1.confidence_percentage + "%");

                // Actualizar resultados de Model 2
                $('#model2-class').text("Clase predicha: " + data.model_2.predicted_class);
                $('#model2-confidence').text("Confianza: " + data.model_2.confidence_percentage + "%");
            },
            error: function () {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html("<p>Hubo un error en la predicción. Inténtalo de nuevo.</p>");
            }
        });
    });
});
